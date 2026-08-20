//
// Copyright 2026 Signal Messenger, LLC
// SPDX-License-Identifier: AGPL-3.0-only
//

use std::{collections::VecDeque, ops::Range};

use calling_common::{Duration, Instant};
use smallvec::SmallVec;
use thiserror::Error;

use crate::rtp::{FullFrameNumber, FullSequenceNumber};

/// Default maximum number of frames in flight. Since we expect minimal frame overlap and that
/// frames need to be processed quickly, this number can be kept low.
pub const MAX_FRAMES_IN_FLIGHT: usize = 10;
/// The dependency descriptor allows frames to refer to frames that were seen 4096 frames "ago".
/// We therefore default to 4096 here, which is unlikely.
pub const DEFAULT_COMPLETE_FRAMES_TRACKED: usize = 4096;
/// Default period of time between two consecutive calls to `FrameTracker::do_periodic_cleanup`
/// that will result in cleanup being performed.
pub const DEFAULT_PRUNE_PERIOD: Duration = Duration::from_millis(500);
/// Default period of time a frame is allowed to remain in the "frame-in-flight" state before
/// it is discarded.
pub const DEFAULT_FRAME_LIFETIME: Duration = Duration::from_secs(5);

#[derive(Error, Debug, PartialEq, Eq)]
pub enum FrameTrackerError {
    #[error("Start flag already set for frame {0}")]
    FrameStartFlagAlreadySet(FullFrameNumber),
    #[error("End flag already set for frame {0}")]
    FrameEndFlagAlreadySet(FullFrameNumber),
    #[error("Too many missing packet ranges for frame {0}")]
    TooManyMissingPacketRanges(FullFrameNumber),
}

/// Instances of `FrameTracker` are used to track which frames have been fully received
/// for a particular RTP stream. A frame that has been fully received is considered to
/// be *complete* and has the following characteristics:
///  - It has a start seqnum
///  - It has an end seqnum
///  - It has no missed packets
///
/// A frame that is being tracked but is not yet in the *complete* state is considered
/// to be a frame in flight. A frame can be a frame in flight only for a relatively short
/// period. After that period expires, the frame is purged and is considered lost.
///
/// Please see `FrameTrackerConfig` for information on how to configure a `FrameTracker`
/// instance.
///
/// **IMPORTANT**
///
/// The frame tracker assumes that it is communicating with a WebRTC endpoint. Consequently,
/// it expects that there will be no frame overlap. In other words, all packets received
/// between the first and the last packets of a frame will belong to that same frame.
#[derive(Debug)]
pub struct FrameTracker {
    max_complete_frames: usize,
    complete_frames: VecDeque<FullFrameNumber>,
    frames_in_flight: SmallVec<[FrameInfo; MAX_FRAMES_IN_FLIGHT]>,
    next_prune_time: Instant,
    prune_period: Duration,
    frame_lifetime: Duration,
}

impl Default for FrameTracker {
    fn default() -> Self {
        Self::new(FrameTrackerConfig::default())
    }
}

pub struct FrameTrackerConfig {
    /// Maximum number of frame numbers identifying complete frames to store. This list
    /// is maintained in the FIFO fashion.
    pub max_complete_frames: usize,
    /// How frequently to perform the pruning operation that discards stale frames in flight.
    pub prune_period: Duration,
    /// How long a frame is allowed to remain in the "frame-in-flight" state before it
    /// is discarded.
    pub frame_lifetime: Duration,
}

impl Default for FrameTrackerConfig {
    fn default() -> Self {
        Self {
            max_complete_frames: DEFAULT_COMPLETE_FRAMES_TRACKED,
            prune_period: DEFAULT_PRUNE_PERIOD,
            frame_lifetime: DEFAULT_FRAME_LIFETIME,
        }
    }
}

#[derive(Debug, Default)]
pub struct PacketInfo {
    pub seqnum: FullSequenceNumber,
    pub frame_number: FullFrameNumber,
    pub start_frame_flag: bool,
    pub end_frame_flag: bool,
}

impl FrameTracker {
    pub fn new(config: FrameTrackerConfig) -> Self {
        let FrameTrackerConfig {
            max_complete_frames,
            prune_period,
            frame_lifetime,
        } = config;
        Self {
            max_complete_frames,
            prune_period,
            frame_lifetime,
            complete_frames: VecDeque::new(),
            frames_in_flight: SmallVec::new(),
            next_prune_time: Instant::now() + prune_period,
        }
    }

    fn push_complete_frame(&mut self, frame_number: FullFrameNumber) {
        // Guard against a pathological case where the max_complete_frames is set to 0.
        // We should probably disallow this case in the future.
        if self.max_complete_frames == 0 {
            return;
        }
        if self.complete_frames.len() >= self.max_complete_frames {
            self.complete_frames.pop_front();
        }
        // Find an appropriate place for the frame number so that the list remains sorted.
        // Start scanning from the back as the insertion point will generally be at the very
        // end of the list or very close to it.
        if let Some(index) = self.complete_frames.iter().rposition(|v| *v < frame_number) {
            self.complete_frames.insert(index + 1, frame_number);
        } else {
            self.complete_frames.push_front(frame_number);
        }
    }

    /// Returns `true` if the frame with the given frame number can be considered *complete*.
    /// A complete frame is the one that has a start seqnum, end seqnum, and for which
    /// there are no missed packets.
    pub fn is_complete(&self, frame_number: FullFrameNumber) -> bool {
        const EXPECTED_REFERENCE_RANGE: usize = 10;

        // We expect the frame to be close to the end of the list. We do a reverse
        // sequential scan over the expected reference range at the end of the list.
        // If the frame is not found, we do a full binary search.
        self.complete_frames
            .iter()
            .rev()
            .take(EXPECTED_REFERENCE_RANGE)
            .any(|n| *n == frame_number)
            || self.complete_frames.binary_search(&frame_number).is_ok()
    }

    pub fn update(
        &mut self,
        now: Instant,
        packet_info: PacketInfo,
    ) -> Result<(), FrameTrackerError> {
        let PacketInfo { frame_number, .. } = packet_info;

        if let Some(index) = self
            .frames_in_flight
            .iter()
            .position(|frame| frame.frame_number == frame_number)
        {
            let frame = &mut self.frames_in_flight[index];
            if let Err(e) = frame.handle_packet(packet_info) {
                // Drop the frame if the number of missing seqnum ranges exceeds the limit.
                if matches!(e, FrameTrackerError::TooManyMissingPacketRanges(_)) {
                    self.frames_in_flight.swap_remove(index);
                }
                return Err(e);
            }
            if frame.is_complete() {
                self.push_complete_frame(frame_number);
                self.frames_in_flight.swap_remove(index);
            }
        } else {
            let expires_at = now + self.frame_lifetime;
            let mut frame = FrameInfo::new(frame_number, expires_at);
            frame.handle_packet(packet_info)?;
            if frame.is_complete() {
                self.push_complete_frame(frame_number);
            } else {
                self.frames_in_flight.push(frame);
            }
        }

        Ok(())
    }

    /// Performs periodic cleanup tasks. This should be invoked periodically to release
    /// resources that are considered expired.
    pub fn do_periodic_cleanup(&mut self, now: Instant) {
        if now >= self.next_prune_time {
            self.next_prune_time = now + self.prune_period;
            self.frames_in_flight.retain(|frame| frame.expires_at > now);
        }
    }

    /// Returns the number of frames in flight that are being tracked.
    #[must_use]
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.frames_in_flight.len()
    }

    /// Returns `true` if there are no frames in flight.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.frames_in_flight.is_empty()
    }
}

#[derive(Debug)]
struct FrameInfo {
    expires_at: Instant,
    frame_number: FullFrameNumber,
    start_seqnum: Option<FullSequenceNumber>,
    end_seqnum: Option<FullSequenceNumber>,
    min_seqnum: Option<FullSequenceNumber>,
    max_seqnum: Option<FullSequenceNumber>,
    // Missing seqnums ranges are used to track missing packets. We allow up to 8 values
    // within `FrameInfo` before we start spilling to heap. The overall maximum is
    // controlled by `FrameInfo::MAX_MISSING_SEQNUM_RANGES`.
    missing_seqnum_ranges: SmallVec<[Range<FullSequenceNumber>; 8]>,
}

impl FrameInfo {
    // Limit the maximum number of missing seqnums ranges to 20.
    const MAX_MISSING_SEQNUM_RANGES: usize = 20;

    fn new(frame_number: FullFrameNumber, expires_at: Instant) -> Self {
        Self {
            expires_at,
            frame_number,
            start_seqnum: None,
            end_seqnum: None,
            min_seqnum: None,
            max_seqnum: None,
            missing_seqnum_ranges: SmallVec::new(),
        }
    }

    #[inline]
    fn is_complete(&self) -> bool {
        self.missing_seqnum_ranges.is_empty()
            && self.start_seqnum.is_some()
            && self.end_seqnum.is_some()
    }

    #[inline]
    fn push_missing_seqnum_range_if_not_empty(
        &mut self,
        range: Range<FullSequenceNumber>,
    ) -> Result<(), FrameTrackerError> {
        if !range.is_empty() {
            if self.missing_seqnum_ranges.len() >= Self::MAX_MISSING_SEQNUM_RANGES {
                return Err(FrameTrackerError::TooManyMissingPacketRanges(
                    self.frame_number,
                ));
            }
            self.missing_seqnum_ranges.push(range);
        }
        Ok(())
    }

    fn handle_packet(&mut self, packet_info: PacketInfo) -> Result<(), FrameTrackerError> {
        let PacketInfo {
            start_frame_flag,
            end_frame_flag,
            seqnum,
            ..
        } = packet_info;

        if start_frame_flag && self.start_seqnum.is_some() {
            return Err(FrameTrackerError::FrameStartFlagAlreadySet(
                self.frame_number,
            ));
        }
        if end_frame_flag && self.end_seqnum.is_some() {
            return Err(FrameTrackerError::FrameEndFlagAlreadySet(self.frame_number));
        }
        if start_frame_flag {
            self.start_seqnum = Some(seqnum);
        }
        if end_frame_flag {
            self.end_seqnum = Some(seqnum);
        }
        if let Some(min_seqnum) = self.min_seqnum {
            if seqnum < min_seqnum {
                self.push_missing_seqnum_range_if_not_empty(seqnum + 1..min_seqnum)?;
                self.min_seqnum = Some(seqnum);
                if self.max_seqnum.is_none() {
                    self.max_seqnum = Some(seqnum);
                }
                return Ok(());
            }
        } else {
            self.min_seqnum = Some(seqnum);
        }
        if let Some(max_seqnum) = self.max_seqnum {
            // We are dealing with extended sequence numbers. Theoretically, wraparound is
            // possible but extremely unlikely.
            if seqnum < max_seqnum {
                if let Some(range_index) = self
                    .missing_seqnum_ranges
                    .iter()
                    .position(|r| r.contains(&seqnum))
                {
                    // Remove the range and create additional ranges if necessary.
                    let range = self.missing_seqnum_ranges.swap_remove(range_index);
                    self.push_missing_seqnum_range_if_not_empty(range.start..seqnum)?;
                    self.push_missing_seqnum_range_if_not_empty(seqnum + 1..range.end)?;
                } else {
                    self.push_missing_seqnum_range_if_not_empty(seqnum + 1..max_seqnum)?;
                }
            } else {
                if max_seqnum + 1 < seqnum {
                    self.push_missing_seqnum_range_if_not_empty(max_seqnum + 1..seqnum)?;
                }
                self.max_seqnum = Some(seqnum);
            }
        } else {
            self.max_seqnum = Some(seqnum);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::svc::frame_tracker::*;

    #[test]
    fn test_handle_packet() -> Result<(), FrameTrackerError> {
        let config = FrameTrackerConfig::default();
        let mut frame_tracker = FrameTracker::new(config);

        let now = Instant::now();

        frame_tracker.update(
            now,
            PacketInfo {
                frame_number: 0,
                start_frame_flag: true,
                end_frame_flag: false,
                seqnum: 0,
            },
        )?;
        frame_tracker.update(
            now,
            PacketInfo {
                frame_number: 0,
                start_frame_flag: false,
                end_frame_flag: false,
                seqnum: 3,
            },
        )?;
        frame_tracker.update(
            now,
            PacketInfo {
                frame_number: 0,
                start_frame_flag: false,
                end_frame_flag: true,
                seqnum: 4,
            },
        )?;

        assert!(!frame_tracker.is_complete(0));

        frame_tracker.update(
            now,
            PacketInfo {
                frame_number: 0,
                start_frame_flag: false,
                end_frame_flag: false,
                seqnum: 2,
            },
        )?;
        frame_tracker.update(
            now,
            PacketInfo {
                frame_number: 0,
                start_frame_flag: false,
                end_frame_flag: false,
                seqnum: 1,
            },
        )?;

        assert!(frame_tracker.is_complete(0));

        Ok(())
    }

    #[test]
    fn test_minseq() -> Result<(), FrameTrackerError> {
        let config = FrameTrackerConfig::default();
        let mut frame_tracker = FrameTracker::new(config);

        let now = Instant::now();

        frame_tracker.update(
            now,
            PacketInfo {
                frame_number: 0,
                start_frame_flag: false,
                end_frame_flag: false,
                seqnum: 2,
            },
        )?;
        frame_tracker.update(
            now,
            PacketInfo {
                frame_number: 0,
                start_frame_flag: false,
                end_frame_flag: true,
                seqnum: 3,
            },
        )?;
        frame_tracker.update(
            now,
            PacketInfo {
                frame_number: 0,
                start_frame_flag: true,
                end_frame_flag: false,
                seqnum: 0,
            },
        )?;
        frame_tracker.update(
            now,
            PacketInfo {
                frame_number: 0,
                start_frame_flag: false,
                end_frame_flag: false,
                seqnum: 1,
            },
        )?;

        assert!(frame_tracker.is_complete(0));

        Ok(())
    }

    #[test]
    fn test_single_frame() -> Result<(), FrameTrackerError> {
        let now = Instant::now();
        let mut frames = FrameTracker::new(FrameTrackerConfig::default());
        frames.update(
            now,
            PacketInfo {
                frame_number: 1,
                start_frame_flag: true,
                end_frame_flag: false,
                seqnum: 1,
            },
        )?;
        assert_eq!(frames.len(), 1);
        frames.update(
            now,
            PacketInfo {
                frame_number: 1,
                start_frame_flag: false,
                end_frame_flag: true,
                seqnum: 2,
            },
        )?;
        assert!(frames.is_empty());

        Ok(())
    }

    #[test]
    fn test_multiple_frames() -> Result<(), FrameTrackerError> {
        let now = Instant::now();
        const FRAME_COUNT: FullFrameNumber = 10;
        let mut frames = FrameTracker::new(FrameTrackerConfig::default());
        let mut seqnum = 0;
        for i in 0..FRAME_COUNT {
            frames.update(
                now,
                PacketInfo {
                    frame_number: i,
                    start_frame_flag: true,
                    end_frame_flag: false,
                    seqnum,
                },
            )?;
            seqnum += 3;
        }
        seqnum = 2;
        for i in 0..FRAME_COUNT {
            frames.update(
                now,
                PacketInfo {
                    frame_number: i,
                    start_frame_flag: false,
                    end_frame_flag: true,
                    seqnum,
                },
            )?;
            seqnum += 3;
        }
        seqnum = 1;
        for i in 0..FRAME_COUNT {
            frames.update(
                now,
                PacketInfo {
                    frame_number: i,
                    start_frame_flag: false,
                    end_frame_flag: false,
                    seqnum,
                },
            )?;
            seqnum += 3;
        }
        assert!(frames.is_empty());

        Ok(())
    }

    #[test]
    fn test_single_packet_frame() -> Result<(), FrameTrackerError> {
        let now = Instant::now();
        let mut tracker = FrameTracker::new(FrameTrackerConfig::default());
        tracker.update(
            now,
            PacketInfo {
                frame_number: 1,
                start_frame_flag: true,
                end_frame_flag: true,
                seqnum: 42,
            },
        )?;
        assert!(tracker.is_empty());
        assert!(tracker.is_complete(1));
        Ok(())
    }

    #[test]
    fn test_duplicate_start_flag_returns_error() {
        let now = Instant::now();
        let mut tracker = FrameTracker::new(FrameTrackerConfig::default());
        tracker
            .update(
                now,
                PacketInfo {
                    frame_number: 1,
                    start_frame_flag: true,
                    end_frame_flag: false,
                    seqnum: 1,
                },
            )
            .unwrap();
        let result = tracker.update(
            now,
            PacketInfo {
                frame_number: 1,
                start_frame_flag: true,
                end_frame_flag: false,
                seqnum: 2,
            },
        );
        assert_eq!(result, Err(FrameTrackerError::FrameStartFlagAlreadySet(1)));
    }

    #[test]
    fn test_duplicate_end_flag_returns_error() {
        let now = Instant::now();
        let mut tracker = FrameTracker::new(FrameTrackerConfig::default());
        tracker
            .update(
                now,
                PacketInfo {
                    frame_number: 1,
                    start_frame_flag: false,
                    end_frame_flag: true,
                    seqnum: 1,
                },
            )
            .unwrap();
        let result = tracker.update(
            now,
            PacketInfo {
                frame_number: 1,
                start_frame_flag: false,
                end_frame_flag: true,
                seqnum: 2,
            },
        );
        assert_eq!(result, Err(FrameTrackerError::FrameEndFlagAlreadySet(1)));
    }

    #[test]
    fn test_too_many_missing_packets_returns_error() {
        let now = Instant::now();
        let mut tracker = FrameTracker::new(FrameTrackerConfig::default());
        // Send packets at even seqnums (0, 2, 4, ...) to create one gap per step.
        // After 20 gaps (MAX_MISSING_SEQNUM_RANGES), the next gap triggers an error.
        let mut seqnum = 0;
        tracker
            .update(
                now,
                PacketInfo {
                    frame_number: 0,
                    start_frame_flag: true,
                    end_frame_flag: false,
                    seqnum,
                },
            )
            .unwrap();
        for _ in 0..20 {
            seqnum += 2;
            tracker
                .update(
                    now,
                    PacketInfo {
                        frame_number: 0,
                        start_frame_flag: false,
                        end_frame_flag: false,
                        seqnum,
                    },
                )
                .unwrap();
        }
        seqnum += 2;
        let result = tracker.update(
            now,
            PacketInfo {
                frame_number: 0,
                start_frame_flag: false,
                end_frame_flag: false,
                seqnum,
            },
        );
        assert_eq!(
            result,
            Err(FrameTrackerError::TooManyMissingPacketRanges(0))
        );
    }

    #[test]
    fn test_frame_expiry_via_periodic_cleanup() {
        let now = Instant::now();
        let mut tracker = FrameTracker::new(FrameTrackerConfig::default());
        tracker
            .update(
                now,
                PacketInfo {
                    frame_number: 1,
                    start_frame_flag: true,
                    end_frame_flag: false,
                    seqnum: 1,
                },
            )
            .unwrap();
        assert_eq!(tracker.len(), 1);

        // Advance past frame_lifetime (5s) and prune_period (500ms).
        let expired = now + DEFAULT_FRAME_LIFETIME + Duration::from_millis(1);
        tracker.do_periodic_cleanup(expired);
        assert!(tracker.is_empty());
        assert!(!tracker.is_complete(1));
    }

    #[test]
    fn test_max_complete_frames_evicts_oldest() -> Result<(), FrameTrackerError> {
        let config = FrameTrackerConfig {
            max_complete_frames: 3,
            prune_period: DEFAULT_PRUNE_PERIOD,
            frame_lifetime: DEFAULT_FRAME_LIFETIME,
        };
        let now = Instant::now();
        let mut tracker = FrameTracker::new(config);
        for i in 0..4 {
            tracker.update(
                now,
                PacketInfo {
                    frame_number: i,
                    start_frame_flag: true,
                    end_frame_flag: true,
                    seqnum: i,
                },
            )?;
        }
        assert!(!tracker.is_complete(0)); // evicted
        assert!(tracker.is_complete(1));
        assert!(tracker.is_complete(2));
        assert!(tracker.is_complete(3));
        Ok(())
    }

    #[test]
    fn test_is_complete_binary_search_fallback() -> Result<(), FrameTrackerError> {
        let now = Instant::now();
        let mut tracker = FrameTracker::new(FrameTrackerConfig::default());
        // Complete 15 frames. Frame 0 is 14 positions from the end of complete_frames,
        // past the 10-element linear scan, so binary search is exercised.
        for i in 0..15 {
            tracker.update(
                now,
                PacketInfo {
                    frame_number: i,
                    start_frame_flag: true,
                    end_frame_flag: true,
                    seqnum: i,
                },
            )?;
        }
        assert!(tracker.is_complete(0)); // found via binary search
        assert!(tracker.is_complete(14)); // found via linear scan
        Ok(())
    }

    #[test]
    fn test_duplicate_seqnum_is_ignored() -> Result<(), FrameTrackerError> {
        let now = Instant::now();
        let mut tracker = FrameTracker::new(FrameTrackerConfig::default());
        tracker.update(
            now,
            PacketInfo {
                frame_number: 1,
                start_frame_flag: true,
                end_frame_flag: false,
                seqnum: 5,
            },
        )?;
        // Same seqnum should not error.
        tracker.update(
            now,
            PacketInfo {
                frame_number: 1,
                start_frame_flag: false,
                end_frame_flag: false,
                seqnum: 5,
            },
        )?;
        assert_eq!(tracker.len(), 1);
        tracker.update(
            now,
            PacketInfo {
                frame_number: 1,
                start_frame_flag: false,
                end_frame_flag: true,
                seqnum: 6,
            },
        )?;
        assert!(tracker.is_empty());
        assert!(tracker.is_complete(1));
        Ok(())
    }

    #[test]
    fn test_gap_recorded_when_packet_arrives_below_max_seqnum() -> Result<(), FrameTrackerError> {
        // When the first-received packet sets max_seqnum above the start packet's seqnum,
        // the gap between them should be tracked. Packet arrival order: 5, 1 (start), 7 (end), 6.
        // Seqnums 2, 3, 4 are never received.
        let now = Instant::now();
        let mut tracker = FrameTracker::new(FrameTrackerConfig::default());

        // Seqnum 5 arrives first — sets max_seqnum=5 with no gaps below it tracked.
        tracker.update(
            now,
            PacketInfo {
                frame_number: 1,
                start_frame_flag: false,
                end_frame_flag: false,
                seqnum: 5,
            },
        )?;

        // Start packet at seqnum 1: below max_seqnum, gap [1..5) is recorded.
        // Note: seqnum 1 itself is included in the missing range even though it just arrived;
        // the range should ideally be (seqnum+1)..max_seqnum.
        tracker.update(
            now,
            PacketInfo {
                frame_number: 1,
                start_frame_flag: true,
                end_frame_flag: false,
                seqnum: 1,
            },
        )?;

        // End packet at seqnum 7: gap [6..7) is tracked above max_seqnum.
        tracker.update(
            now,
            PacketInfo {
                frame_number: 1,
                start_frame_flag: false,
                end_frame_flag: true,
                seqnum: 7,
            },
        )?;

        // Fill the gap above the original max_seqnum. Seqnums 2, 3, 4 were never received.
        tracker.update(
            now,
            PacketInfo {
                frame_number: 1,
                start_frame_flag: false,
                end_frame_flag: false,
                seqnum: 6,
            },
        )?;

        // Frame is not complete: the untracked gap is now recorded, preventing false completion.
        assert!(!tracker.is_complete(1));
        Ok(())
    }
}
