//
// Copyright 2026 Signal Messenger, LLC
// SPDX-License-Identifier: AGPL-3.0-only
//

use std::cmp::{max, min};

use calling_common::{DataRate, DemuxId, Duration, Instant, VideoHeight};
use log::warn;
use smallvec::{SmallVec, smallvec};
use thiserror::Error;

use crate::{
    rtp::MAX_DECODE_TARGETS,
    svc::{
        DecodeTarget, DecodeTargetInfo, DecodeTargetInfoList, DecodeTargetInfoLists,
        MAX_EXPECTED_CLIENTS,
    },
};

pub type SelectedDecodeTargets = SmallVec<[Option<DecodeTarget>; MAX_EXPECTED_CLIENTS]>;

/// A structure representing the result of an allocation process.
#[derive(Debug, Default, PartialEq)]
pub struct AllocationResult {
    /// The bandwidth that has been allocated.
    pub allocated_rate: DataRate,
    /// Collection of optional `DecodeTarget` values, one for each sender.
    pub selected_decode_targets: SelectedDecodeTargets,
}

#[derive(Debug, Error, PartialEq)]
pub enum AllocationError {
    #[error("Mismatched height constraints")]
    MismatchedHeightConstraints,
}

pub trait Allocator {
    /// Allocates resources for a given data rate, optional video heights, and decode target
    /// information lists.
    ///
    /// # Parameters
    ///
    /// - `rate_budget`: The `DataRate` parameter specifying the allocation budget to be distributed.
    /// - `heights`: An optional slice of `VideoHeight` values, representing the set of heights
    ///   to be considered during allocation.
    ///   - If provided, allocation will consider these specific video heights.
    ///   - If `None`, allocation will consider all available video heights.
    /// - `decode_target_info_lists`: A slice of references to `DecodeTargetInfoList`s, which
    ///   define the decoding target requirements that
    ///   guide the allocation process.
    ///
    /// # Returns
    ///
    /// - `Ok(AllocationResult)`: Contains the results of the allocation process if successful,
    ///   including any allocations made.
    /// - `Err(AllocationError)`: Indicates an error occurred during allocation, such as
    ///   insufficient resources or invalid parameters.
    fn allocate(
        &self,
        rate_budget: DataRate,
        heights: Option<&[VideoHeight]>,
        decode_target_info_lists: &[&DecodeTargetInfoList],
    ) -> Result<AllocationResult, AllocationError>;

    fn calculate_ideal_send_rate(
        &self,
        heights: Option<&[VideoHeight]>,
        decode_target_info_lists: &[&DecodeTargetInfoList],
    ) -> Result<AllocationResult, AllocationError>;

    fn calculate_base_rate(&self, decode_target_info_lists: &[&DecodeTargetInfoList]) -> DataRate;
}

/// [`DefaultAllocator`] implements a greedy bandwidth allocation strategy.  It iterates through
/// decode targets tier-by-tier, greedily assigning higher tiers to senders as long as
/// the running total within each tier fits the budget.
#[derive(Debug, Default)]
pub struct DefaultAllocator;

impl DefaultAllocator {
    /// Allocates a base layer from the provided decode target information lists while
    /// adhering to the given rate budget.
    ///
    /// # Parameters
    /// - `rate_budget`: The maximum allowable data rate for allocation.
    /// - `target_info_lists`: A slice of references to `DecodeTargetInfoList`. Each list
    ///   contains information about the available decode targets for a specific sender.
    /// - `selected_targets`: A mutable collection of `SelectedDecodeTargets` where
    ///   the selected target indices for each sender will be updated if allocation is successful.
    /// - `predicate`: A mutable closure that is used to filter the decode targets.
    ///   It takes the following arguments:
    ///   - `DecodeTarget`: The decode target being evaluated.
    ///   - `usize`: The index of the sender being considered.
    ///   - `&DecodeTargetInfo`: The information of the current decode target to evaluate.
    ///     The closure should return `true` if the decode target can be considered for allocation.
    ///
    /// # Returns
    /// A tuple containing:
    /// - `bool`: `true` if the allocation was successful without exceeding the rate budget;
    ///   `false` otherwise.
    /// - `DataRate`: The total data rate successfully allocated.
    fn allocate_base_layer<F>(
        &self,
        rate_budget: DataRate,
        target_info_lists: &[&DecodeTargetInfoList],
        selected_targets: &mut SelectedDecodeTargets,
        predicate: &mut F,
    ) -> (bool, DataRate)
    where
        F: FnMut(DecodeTarget, usize, &DecodeTargetInfo) -> bool,
    {
        debug_assert_eq!(target_info_lists.len(), selected_targets.len());

        let sender_count = target_info_lists.len();
        let mut success = true;
        let mut allocated_rate = DataRate::ZERO;

        for i in 0..sender_count {
            let info_list = target_info_lists[i];
            if !info_list.is_empty() && predicate(0, i, &info_list[0]) {
                let possible_rate = allocated_rate + info_list[0].rate;
                if possible_rate > rate_budget {
                    success = false;
                    break;
                }
                allocated_rate = possible_rate;
                selected_targets[i] = Some(0);
                if allocated_rate == rate_budget {
                    break;
                }
            }
        }

        (success, allocated_rate)
    }

    /// Attempts to allocate data rates to the upper layers of decode targets while
    /// considering a budget, constraints, and a predicate for selecting targets based
    /// on specific criteria. This function makes use of a greedy approach to optimize
    /// the selection process.
    ///
    /// # Arguments
    ///
    /// - `rate_budget`: The maximum allowable data rate to allocate. Once this budget
    ///   is surpassed, the function will return the currently allocated rate.
    /// - `target_info_lists`: A slice of references to lists containing decode target
    ///   information. Each list corresponds to a sender and provides details about
    ///   potential decode targets.
    /// - `max_targets`: The maximum number of decode targets to consider for allocation.
    /// - `allocated_rate`: The current rate that has already been allocated before
    ///   calling this function.
    /// - `selected_targets`: A mutable reference to a container holding the indices of
    ///   the currently selected decode targets for each sender. This container will be
    ///   updated to reflect newly selected targets that honor the constraints of `rate_budget`
    ///   and `predicate`.
    /// - `predicate`: A mutable closure that determines if a specific decode target is eligible
    ///   for allocation. It is provided with the following parameters:
    ///    - `DecodeTarget`: The current decode target being evaluated.
    ///    - `usize`: The index of the sender being processed.
    ///    - `DecodeTargetInfo`: Information about the decode target.
    ///
    /// # Returns
    ///
    /// The total data rate allocated after evaluating all decode targets up to `max_targets`.
    /// If allocating a new target would exceed the `rate_budget`, the previously allocated
    /// data rate is returned.
    fn allocate_upper_layers<F>(
        &self,
        rate_budget: DataRate,
        target_info_lists: &[&DecodeTargetInfoList],
        max_targets: usize,
        allocated_rate: DataRate,
        selected_targets: &mut SelectedDecodeTargets,
        predicate: &mut F,
    ) -> DataRate
    where
        F: FnMut(DecodeTarget, usize, &DecodeTargetInfo) -> bool,
    {
        debug_assert_eq!(target_info_lists.len(), selected_targets.len());

        let sender_count = target_info_lists.len();
        let mut tentative_targets = selected_targets.clone();
        let mut allocated_rate = allocated_rate;

        for target in 1..max_targets {
            let mut tentative_rate = DataRate::ZERO;
            for i in 0..sender_count {
                let list = target_info_lists[i];
                let rate_info = match list.get(target) {
                    Some(target_info) if predicate(target, i, target_info) => {
                        Some((target_info.rate, target))
                    }
                    _ => selected_targets[i]
                        .map(|selected_target| (list[selected_target].rate, selected_target)),
                };
                if let Some((rate, selected_target)) = rate_info {
                    tentative_rate = tentative_rate + rate;
                    if tentative_rate > rate_budget {
                        return allocated_rate;
                    }
                    tentative_targets[i] = Some(selected_target);
                }
            }
            allocated_rate = tentative_rate;
            selected_targets.copy_from_slice(&tentative_targets);
        }

        allocated_rate
    }

    /// Allocates data rate across multiple decode targets based on the provided rate budget
    /// and filtering logic.
    ///
    /// # Parameters
    ///
    /// - `rate_budget`: The total `DataRate` budget that can be allocated among
    ///   the decode targets.
    /// - `decode_target_info_lists`: A slice of references to `DecodeTargetInfoList` instances,
    ///   where each list contains information about the decode targets associated with a layer.
    /// - `filter`: A closure of type `F`, which is called for every combination of
    ///   `DecodeTarget`, index, and associated `DecodeTargetInfo`.
    ///   The closure should return a boolean indicating whether the specified decode target
    ///   should be considered for allocation.
    ///
    /// # Returns
    ///
    /// Returns an `AllocationResult` containing:
    /// - `allocated_rate`: The total `DataRate` that was successfully allocated in this operation.
    /// - `selected_decode_targets`: A collection of selected decode target indices after the
    ///   allocation process.
    fn do_allocate<F>(
        &self,
        rate_budget: DataRate,
        decode_target_info_lists: &[&DecodeTargetInfoList],
        mut filter: F,
    ) -> AllocationResult
    where
        F: FnMut(DecodeTarget, usize, &DecodeTargetInfo) -> bool,
    {
        let max_targets = decode_target_info_lists
            .iter()
            .map(|list| list.len())
            .max()
            .unwrap_or(0);
        debug_assert!(max_targets <= MAX_DECODE_TARGETS);
        let mut selected_decode_targets = smallvec![None; decode_target_info_lists.len()];
        if max_targets == 0 {
            return AllocationResult {
                allocated_rate: DataRate::ZERO,
                selected_decode_targets,
            };
        }
        let (success, base_layer_rate) = self.allocate_base_layer(
            rate_budget,
            decode_target_info_lists,
            &mut selected_decode_targets,
            &mut filter,
        );
        if !success {
            return AllocationResult {
                allocated_rate: base_layer_rate,
                selected_decode_targets,
            };
        }
        let allocated_rate = self.allocate_upper_layers(
            rate_budget,
            decode_target_info_lists,
            max_targets,
            base_layer_rate,
            &mut selected_decode_targets,
            &mut filter,
        );
        AllocationResult {
            allocated_rate,
            selected_decode_targets,
        }
    }
}

impl Allocator for DefaultAllocator {
    fn allocate(
        &self,
        rate_budget: DataRate,
        heights: Option<&[VideoHeight]>,
        decode_target_info_lists: &[&DecodeTargetInfoList],
    ) -> Result<AllocationResult, AllocationError> {
        match heights {
            Some(heights) if heights.len() != decode_target_info_lists.len() => {
                Err(AllocationError::MismatchedHeightConstraints)
            }
            Some(heights) => Ok(self.do_allocate(
                rate_budget,
                decode_target_info_lists,
                |_, i, target_info| {
                    target_info.resolution.height <= heights[i].as_u16()
                        && target_info.rate > DataRate::ZERO
                },
            )),
            _ => Ok(self.do_allocate(
                rate_budget,
                decode_target_info_lists,
                |_, _, target_info| target_info.rate > DataRate::ZERO,
            )),
        }
    }

    fn calculate_ideal_send_rate(
        &self,
        heights: Option<&[VideoHeight]>,
        decode_target_info_lists: &[&DecodeTargetInfoList],
    ) -> Result<AllocationResult, AllocationError> {
        match heights {
            Some(heights) if heights.len() != decode_target_info_lists.len() => {
                Err(AllocationError::MismatchedHeightConstraints)
            }
            Some(heights) => Ok(self.do_allocate(
                DataRate::MAX,
                decode_target_info_lists,
                |_, i, target_info| {
                    target_info.resolution.height <= heights[i].as_u16()
                        && target_info.rate > DataRate::ZERO
                },
            )),
            _ => Ok(self.do_allocate(
                DataRate::MAX,
                decode_target_info_lists,
                |_, _, target_info| target_info.rate > DataRate::ZERO,
            )),
        }
    }

    fn calculate_base_rate(&self, decode_target_info_lists: &[&DecodeTargetInfoList]) -> DataRate {
        decode_target_info_lists
            .iter()
            .fold(DataRate::ZERO, |total_rate, list| {
                total_rate + list.first().map(|info| info.rate).unwrap_or(DataRate::ZERO)
            })
    }
}

/// Simple allocator based on the default allocator that disregards the rate budget.
/// Useful only for testing and troubleshooting.
#[derive(Default)]
pub struct BlastAllocator {
    allocator: DefaultAllocator,
}

impl Allocator for BlastAllocator {
    fn allocate(
        &self,
        _rate_budget: DataRate,
        heights: Option<&[VideoHeight]>,
        decode_target_info_lists: &[&DecodeTargetInfoList],
    ) -> Result<AllocationResult, AllocationError> {
        self.allocator
            .allocate(DataRate::MAX, heights, decode_target_info_lists)
    }

    fn calculate_ideal_send_rate(
        &self,
        heights: Option<&[VideoHeight]>,
        decode_target_info_lists: &[&DecodeTargetInfoList],
    ) -> Result<AllocationResult, AllocationError> {
        self.allocator
            .calculate_ideal_send_rate(heights, decode_target_info_lists)
    }

    fn calculate_base_rate(&self, decode_target_info_lists: &[&DecodeTargetInfoList]) -> DataRate {
        self.allocator.calculate_base_rate(decode_target_info_lists)
    }
}

/// `ThrottledAllocator` is a simple wrapper around an `Allocator` implementation that allows
/// allocations to be performed only at certain intervals.
pub struct ThrottledAllocator {
    demux_id: DemuxId,
    allocator: Box<dyn Allocator + Send>,
    quiesce_period: Duration,
    next_allocation_time: Option<Instant>,
    acquired: bool,
}

impl ThrottledAllocator {
    pub fn new(
        demux_id: DemuxId,
        allocator: Box<dyn Allocator + Send>,
        quiesce_period: Duration,
    ) -> Self {
        Self {
            demux_id,
            allocator,
            quiesce_period,
            acquired: false,
            next_allocation_time: None,
        }
    }

    pub fn demux_id(&self) -> DemuxId {
        self.demux_id
    }

    pub fn acquire(&mut self, now: Instant, force: bool) -> bool {
        if self.acquired && !force {
            return false;
        }
        self.acquired = self.next_allocation_time.is_none()
            || self.next_allocation_time.is_some_and(|v| v < now)
            || force;
        self.acquired
    }

    pub fn release(&mut self, now: Instant) {
        assert!(self.acquired);
        self.acquired = false;
        self.next_allocation_time = Some(now + self.quiesce_period);
    }
}

impl Allocator for ThrottledAllocator {
    fn allocate(
        &self,
        rate_budget: DataRate,
        heights: Option<&[VideoHeight]>,
        decode_target_info_lists: &[&DecodeTargetInfoList],
    ) -> Result<AllocationResult, AllocationError> {
        assert!(self.acquired);
        self.allocator
            .allocate(rate_budget, heights, decode_target_info_lists)
    }

    fn calculate_ideal_send_rate(
        &self,
        heights: Option<&[VideoHeight]>,
        decode_target_info_lists: &[&DecodeTargetInfoList],
    ) -> Result<AllocationResult, AllocationError> {
        assert!(self.acquired);
        self.allocator
            .calculate_ideal_send_rate(heights, decode_target_info_lists)
    }

    fn calculate_base_rate(&self, decode_target_info_lists: &[&DecodeTargetInfoList]) -> DataRate {
        assert!(self.acquired);
        self.allocator.calculate_base_rate(decode_target_info_lists)
    }
}

pub struct BasicAllocationStrategy<'a> {
    pub demux_id: DemuxId,
    pub target_rate: DataRate,
    pub heights: Option<&'a [VideoHeight]>,
    pub decode_target_lists: DecodeTargetInfoLists<'a>,
    pub outgoing_queue_drain_rate: DataRate,
    pub target_rate_allocation_ratio: f64,
}

pub struct BasicAllocationStrategyResult {
    pub ideal_send_rate: DataRate,
    pub requested_base_rate: DataRate,
    pub allocated_rate: DataRate,
    pub selected_decode_targets: SelectedDecodeTargets,
}

impl<'a> BasicAllocationStrategy<'a> {
    pub fn allocate(&self, allocator: &dyn Allocator) -> BasicAllocationStrategyResult {
        let AllocationResult {
            allocated_rate: ideal_send_rate,
            ..
        } = allocator
            .calculate_ideal_send_rate(None, &self.decode_target_lists)
            .unwrap_or_else(|e| {
                warn!("svc: {:?}: ideal send rate: {e}", self.demux_id);
                AllocationResult::default()
            });

        let allocatable_rate = min(
            ideal_send_rate,
            max(
                self.target_rate
                    .saturating_sub(self.outgoing_queue_drain_rate),
                self.target_rate * self.target_rate_allocation_ratio,
            ),
        );

        let AllocationResult {
            allocated_rate,
            selected_decode_targets,
        } = allocator
            .allocate(allocatable_rate, None, &self.decode_target_lists)
            .unwrap_or_else(|e| {
                warn!("svc: {:?}: allocate: {e}", self.demux_id);
                AllocationResult::default()
            });

        let requested_base_rate = allocator.calculate_base_rate(&self.decode_target_lists);

        BasicAllocationStrategyResult {
            ideal_send_rate,
            requested_base_rate,
            allocated_rate,
            selected_decode_targets,
        }
    }
}

#[cfg(test)]
mod tests {
    use calling_common::DataRate;
    use smallvec::smallvec;

    use crate::{
        rtp::Resolution,
        svc::{
            DecodeTargetInfo, DecodeTargetInfoList,
            allocator::{
                AllocationError, AllocationResult, Allocator, DefaultAllocator,
                SelectedDecodeTargets,
            },
        },
    };

    #[test]
    fn test_default_allocator() {
        let client_1_decode_target_info: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(1000),
                resolution: Resolution::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(2000),
                resolution: Default::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(3000),
                resolution: Default::default(),
                chain_index: 0,
            }
        ]
        .into();
        let client_2_decode_target_info: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(512),
                resolution: Resolution::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(1500),
                resolution: Default::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(2500),
                resolution: Default::default(),
                chain_index: 0,
            }
        ]
        .into();
        let decode_target_info_lists = [&client_1_decode_target_info, &client_2_decode_target_info];
        let allocator = DefaultAllocator;
        let allocation_result =
            allocator.allocate(DataRate::from_kbps(10000), None, &decode_target_info_lists);
        let expected_selected_targets: SelectedDecodeTargets = smallvec![Some(2), Some(2)];

        assert_eq!(
            allocation_result,
            Ok(AllocationResult {
                selected_decode_targets: expected_selected_targets,
                allocated_rate: DataRate::from_kbps(5500),
            })
        );
    }

    #[test]
    fn test_default_allocator_with_different_decode_target_counts() {
        let client_1_decode_target_info: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(1000),
                resolution: Resolution::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(2000),
                resolution: Default::default(),
                chain_index: 0,
            },
        ]
        .into();
        let client_2_decode_target_info: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(512),
                resolution: Resolution::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(1500),
                resolution: Default::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(2000),
                resolution: Default::default(),
                chain_index: 0,
            }
        ]
        .into();
        let decode_target_info_lists = [&client_1_decode_target_info, &client_2_decode_target_info];
        let allocator = DefaultAllocator;
        let allocation_result =
            allocator.allocate(DataRate::from_kbps(10000), None, &decode_target_info_lists);
        let expected_selected_targets: SelectedDecodeTargets = smallvec![Some(1), Some(2)];

        assert_eq!(
            allocation_result,
            Ok(AllocationResult {
                selected_decode_targets: expected_selected_targets,
                allocated_rate: DataRate::from_kbps(4000),
            })
        );
    }

    #[test]
    fn test_no_senders() {
        let allocator = DefaultAllocator;
        let result = allocator.allocate(DataRate::from_kbps(10000), None, &[]);
        assert_eq!(
            result,
            Ok(AllocationResult {
                selected_decode_targets: smallvec![],
                allocated_rate: DataRate::ZERO,
            })
        );
    }

    #[test]
    fn test_zero_budget() {
        let client: DecodeTargetInfoList = smallvec![DecodeTargetInfo {
            rate: DataRate::from_kbps(1000),
            resolution: Resolution::default(),
            chain_index: 0,
        }]
        .into();
        let allocator = DefaultAllocator;
        let result = allocator.allocate(DataRate::ZERO, None, &[&client]);
        assert_eq!(
            result,
            Ok(AllocationResult {
                selected_decode_targets: smallvec![None],
                allocated_rate: DataRate::ZERO,
            })
        );
    }

    #[test]
    fn test_single_sender() {
        let client: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(500),
                resolution: Resolution::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(1000),
                resolution: Resolution::default(),
                chain_index: 0,
            },
        ]
        .into();
        let allocator = DefaultAllocator;
        let result = allocator.allocate(DataRate::from_kbps(10000), None, &[&client]);
        assert_eq!(
            result,
            Ok(AllocationResult {
                selected_decode_targets: smallvec![Some(1)],
                allocated_rate: DataRate::from_kbps(1000),
            })
        );
    }

    #[test]
    fn test_budget_caps_at_lowest_tier() {
        // 1000 fits both at tier 0 (500+500=1000) but tier 1 (1001) alone exceeds budget
        let client_1: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(500),
                resolution: Resolution::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(1001),
                resolution: Resolution::default(),
                chain_index: 0,
            },
        ]
        .into();
        let client_2: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(500),
                resolution: Resolution::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(1001),
                resolution: Resolution::default(),
                chain_index: 0,
            },
        ]
        .into();
        let allocator = DefaultAllocator;
        let result = allocator.allocate(DataRate::from_kbps(1000), None, &[&client_1, &client_2]);
        assert_eq!(
            result,
            Ok(AllocationResult {
                selected_decode_targets: smallvec![Some(0), Some(0)],
                allocated_rate: DataRate::from_kbps(1000),
            })
        );
    }

    #[test]
    fn test_tight_budget_first_sender_wins() {
        // 600+600=1200 exceeds budget of 1000, so only sender 0 gets tier 0
        let client_1: DecodeTargetInfoList = smallvec![DecodeTargetInfo {
            rate: DataRate::from_kbps(600),
            resolution: Resolution::default(),
            chain_index: 0,
        }]
        .into();
        let client_2: DecodeTargetInfoList = smallvec![DecodeTargetInfo {
            rate: DataRate::from_kbps(600),
            resolution: Resolution::default(),
            chain_index: 0,
        }]
        .into();
        let allocator = DefaultAllocator;
        let result = allocator.allocate(DataRate::from_kbps(1000), None, &[&client_1, &client_2]);
        assert_eq!(
            result,
            Ok(AllocationResult {
                selected_decode_targets: smallvec![Some(0), None],
                allocated_rate: DataRate::from_kbps(600),
            })
        );
    }

    #[test]
    fn test_zero_rate_decode_target_skips_selection_update() {
        // decode_target[1] has rate=0: selection stays at decode_target[0]
        let client: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(500),
                resolution: Resolution::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::ZERO,
                resolution: Resolution::default(),
                chain_index: 0,
            },
        ]
        .into();
        let allocator = DefaultAllocator;
        let result = allocator.allocate(DataRate::from_kbps(10000), None, &[&client]);
        assert_eq!(
            result,
            Ok(AllocationResult {
                selected_decode_targets: smallvec![Some(0)],
                allocated_rate: DataRate::from_kbps(500),
            })
        );
    }

    #[test]
    fn test_resolution_filter_rejects_over_height_target() {
        // Tier 1 has height 720 which exceeds the 480 limit — should stay at tier 0.
        let client: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(500),
                resolution: Resolution {
                    width: 640,
                    height: 360
                },
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(1000),
                resolution: Resolution {
                    width: 1280,
                    height: 720
                },
                chain_index: 0,
            },
        ]
        .into();
        let allocator = DefaultAllocator;
        let result =
            allocator.allocate(DataRate::from_kbps(10000), Some(&[480.into()]), &[&client]);
        assert_eq!(
            result,
            Ok(AllocationResult {
                selected_decode_targets: smallvec![Some(0)],
                allocated_rate: DataRate::from_kbps(500),
            })
        );
    }

    #[test]
    fn test_upper_tier_requires_all_senders_to_fit() {
        // 900+900=1800 > 1400, so neither sender upgrades even though sender 0 alone would fit.
        let client_1: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(500),
                resolution: Resolution::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(900),
                resolution: Resolution::default(),
                chain_index: 0,
            },
        ]
        .into();
        let client_2: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(500),
                resolution: Resolution::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(900),
                resolution: Resolution::default(),
                chain_index: 0,
            },
        ]
        .into();
        let allocator = DefaultAllocator;
        let result = allocator.allocate(DataRate::from_kbps(1400), None, &[&client_1, &client_2]);
        assert_eq!(
            result,
            Ok(AllocationResult {
                selected_decode_targets: smallvec![Some(0), Some(0)],
                allocated_rate: DataRate::from_kbps(1000),
            })
        );
    }

    #[test]
    fn test_upper_tier_upgrades_all_senders_when_budget_allows() {
        let client_1: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(500),
                resolution: Resolution::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(900),
                resolution: Resolution::default(),
                chain_index: 0,
            },
        ]
        .into();
        let client_2: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(500),
                resolution: Resolution::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(900),
                resolution: Resolution::default(),
                chain_index: 0,
            },
        ]
        .into();
        let allocator = DefaultAllocator;
        let result = allocator.allocate(DataRate::from_kbps(1800), None, &[&client_1, &client_2]);
        assert_eq!(
            result,
            Ok(AllocationResult {
                selected_decode_targets: smallvec![Some(1), Some(1)],
                allocated_rate: DataRate::from_kbps(1800),
            })
        );
    }

    #[test]
    fn test_resolution_filter_with_multiple_senders() {
        // Sender 0's tier 1 (720p) exceeds its 480p limit; sender 1's tier 1 (480p) fits.
        // Sender 0 contributes its tier-0 rate as fallback; tier 1 still commits.
        let client_1: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(500),
                resolution: Resolution {
                    width: 640,
                    height: 360
                },
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(1000),
                resolution: Resolution {
                    width: 1280,
                    height: 720
                },
                chain_index: 0,
            },
        ]
        .into();
        let client_2: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::from_kbps(500),
                resolution: Resolution {
                    width: 640,
                    height: 360
                },
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(600),
                resolution: Resolution {
                    width: 854,
                    height: 480
                },
                chain_index: 0,
            },
        ]
        .into();
        let allocator = DefaultAllocator;
        let result = allocator.allocate(
            DataRate::from_kbps(10000),
            Some(&[480.into(), 480.into()]),
            &[&client_1, &client_2],
        );
        assert_eq!(
            result,
            Ok(AllocationResult {
                selected_decode_targets: smallvec![Some(0), Some(1)],
                allocated_rate: DataRate::from_kbps(1100), // 500 (sender 0 stays at tier 0) + 600 (sender 1 upgrades)
            })
        );
    }

    #[test]
    fn test_default_allocator_with_no_decode_targets() {
        let client_1_decode_target_info: DecodeTargetInfoList = smallvec![].into();
        let client_2_decode_target_info: DecodeTargetInfoList = smallvec![].into();
        let decode_target_info_lists = [&client_1_decode_target_info, &client_2_decode_target_info];
        let allocator = DefaultAllocator;
        let allocation_result =
            allocator.allocate(DataRate::from_kbps(10000), None, &decode_target_info_lists);
        let expected_selected_targets: SelectedDecodeTargets = smallvec![None, None];

        assert_eq!(
            allocation_result,
            Ok(AllocationResult {
                selected_decode_targets: expected_selected_targets,
                allocated_rate: DataRate::ZERO,
            })
        );
    }

    #[test]
    fn test_mismatched_height_constraints() {
        let client: DecodeTargetInfoList = smallvec![DecodeTargetInfo {
            rate: DataRate::from_kbps(500),
            resolution: Resolution::default(),
            chain_index: 0,
        }]
        .into();
        let allocator = DefaultAllocator;
        // 2 height entries for 1 sender → error
        let result = allocator.allocate(
            DataRate::from_kbps(10000),
            Some(&[480.into(), 720.into()]),
            &[&client],
        );
        assert_eq!(result, Err(AllocationError::MismatchedHeightConstraints));
    }

    #[test]
    fn test_zero_rate_tier0_sender_promoted_to_upper_tier() {
        // Tier 0 has rate=0, so it fails the predicate and is skipped at the base layer.
        // Tier 1 has rate>0 and should be picked up during upper-layer allocation.
        let client: DecodeTargetInfoList = smallvec![
            DecodeTargetInfo {
                rate: DataRate::ZERO,
                resolution: Resolution::default(),
                chain_index: 0,
            },
            DecodeTargetInfo {
                rate: DataRate::from_kbps(500),
                resolution: Resolution::default(),
                chain_index: 0,
            },
        ]
        .into();
        let allocator = DefaultAllocator;
        let result = allocator.allocate(DataRate::from_kbps(10000), None, &[&client]);
        assert_eq!(
            result,
            Ok(AllocationResult {
                selected_decode_targets: smallvec![Some(1)],
                allocated_rate: DataRate::from_kbps(500),
            })
        );
    }
}
