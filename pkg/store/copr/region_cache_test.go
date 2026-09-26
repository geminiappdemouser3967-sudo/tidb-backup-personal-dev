// Copyright 2025 PingCAP, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package copr

import (
	"context"
	"testing"
	"time"

	"github.com/pingcap/failpoint"
	"github.com/pingcap/kvproto/pkg/metapb"
	"github.com/pingcap/tidb/pkg/kv"
	"github.com/pingcap/tidb/pkg/store/driver/backoff"
	"github.com/stretchr/testify/require"
	"github.com/tikv/client-go/v2/testutils"
	"github.com/tikv/client-go/v2/tikv"
)

// Test helpers
var (
	// Helper to create KeyRange
	kr = func(start, end string) tikv.KeyRange {
		return tikv.KeyRange{
			StartKey: []byte(start),
			EndKey:   []byte(end),
		}
	}

	// Helper to create KeyLocation
	kl = func(start, end string, regionID uint64) *tikv.KeyLocation {
		return &tikv.KeyLocation{
			Region:   tikv.NewRegionVerID(regionID, 0, 0),
			StartKey: []byte(start),
			EndKey:   []byte(end),
		}
	}
)

// TestValidateLocationCoverage tests various coverage scenarios
func TestValidateLocationCoverage(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name      string
		ranges    []tikv.KeyRange
		locs      []*tikv.KeyLocation
		wantValid bool // whether we expect validation to pass
	}{
		{
			name:      "single range, single location - exact match",
			ranges:    []tikv.KeyRange{kr("a", "z")},
			locs:      []*tikv.KeyLocation{kl("a", "z", 1)},
			wantValid: true,
		},
		{
			name:      "single range, single location - location covers more",
			ranges:    []tikv.KeyRange{kr("b", "y")},
			locs:      []*tikv.KeyLocation{kl("a", "z", 1)},
			wantValid: true,
		},
		{
			name:   "single range split across two locations",
			ranges: []tikv.KeyRange{kr("a", "z")},
			locs: []*tikv.KeyLocation{
				kl("a", "m", 1),
				kl("m", "z", 2),
			},
			wantValid: true, // Valid partial coverage
		},
		{
			name:   "single range split across three locations",
			ranges: []tikv.KeyRange{kr("a", "z")},
			locs: []*tikv.KeyLocation{
				kl("a", "h", 1),
				kl("h", "p", 2),
				kl("p", "z", 3),
			},
			wantValid: true, // Valid partial coverage across multiple locations
		},
		{
			name: "multiple ranges, single location covers all",
			ranges: []tikv.KeyRange{
				kr("b", "d"),
				kr("f", "h"),
			},
			locs:      []*tikv.KeyLocation{kl("a", "z", 1)},
			wantValid: true,
		},
		{
			name: "multiple ranges, multiple locations - aligned",
			ranges: []tikv.KeyRange{
				kr("a", "m"),
				kr("m", "z"),
			},
			locs: []*tikv.KeyLocation{
				kl("a", "m", 1),
				kl("m", "z", 2),
			},
			wantValid: true,
		},
		{
			name: "multiple ranges, multiple locations - disjoint ranges don't require covering gaps",
			ranges: []tikv.KeyRange{
				kr("b", "d"),
				kr("f", "h"),
			},
			locs: []*tikv.KeyLocation{
				kl("a", "d", 1),
				kl("f", "i", 2),
			},
			wantValid: true,
		},
		{
			name: "multiple ranges, multiple locations - overlapping ranges don't require monotonic loc scan",
			ranges: []tikv.KeyRange{
				kr("a", "z"),
				kr("b", "c"),
			},
			locs: []*tikv.KeyLocation{
				kl("a", "m", 1),
				kl("m", "t", 2),
				kl("t", "z", 3),
			},
			wantValid: true,
		},
		{
			name:      "empty start key - location also empty",
			ranges:    []tikv.KeyRange{kr("", "m")},
			locs:      []*tikv.KeyLocation{kl("", "m", 1)},
			wantValid: true,
		},
		{
			name:      "empty start key - location NOT empty",
			ranges:    []tikv.KeyRange{kr("", "m")},
			locs:      []*tikv.KeyLocation{kl("a", "m", 1)},
			wantValid: false, // Invalid - location doesn't start from beginning
		},
		{
			name:      "empty end key - location also empty",
			ranges:    []tikv.KeyRange{kr("m", "")},
			locs:      []*tikv.KeyLocation{kl("m", "", 1)},
			wantValid: true,
		},
		{
			name:      "empty end key - location NOT empty",
			ranges:    []tikv.KeyRange{kr("m", "")},
			locs:      []*tikv.KeyLocation{kl("m", "z", 1)},
			wantValid: false, // Invalid - location doesn't extend to infinity
		},
		{
			name:      "range with empty end - location extends to infinity",
			ranges:    []tikv.KeyRange{kr("m", "")},
			locs:      []*tikv.KeyLocation{kl("a", "", 1)},
			wantValid: true, // Location extends to infinity, covers the range
		},
		{
			name:      "location doesn't cover range start",
			ranges:    []tikv.KeyRange{kr("a", "z")},
			locs:      []*tikv.KeyLocation{kl("b", "z", 1)},
			wantValid: false, // Invalid - location starts after range
		},
		{
			name:      "location doesn't cover range end",
			ranges:    []tikv.KeyRange{kr("a", "z")},
			locs:      []*tikv.KeyLocation{kl("a", "y", 1)},
			wantValid: false, // Invalid - location ends before range and no next location
		},
		{
			name:   "gap between locations",
			ranges: []tikv.KeyRange{kr("a", "z")},
			locs: []*tikv.KeyLocation{
				kl("a", "m", 1),
				kl("n", "z", 2), // Gap between 'm' and 'n'
			},
			wantValid: false, // Invalid - gap in coverage
		},
		{
			name: "discrete ranges with gap between locations - valid",
			ranges: []tikv.KeyRange{
				kr("a", "b"), // First range
				kr("c", "d"), // Second range - discrete, not contiguous
			},
			locs: []*tikv.KeyLocation{
				kl("a", "b", 1), // Covers first range
				kl("c", "d", 2), // Covers second range - gap between locations is OK
			},
			wantValid: true, // Valid - each discrete range is fully covered
		},
		{
			name: "discrete ranges in larger locations with gap - valid",
			ranges: []tikv.KeyRange{
				kr("a", "b"),
				kr("x", "z"),
			},
			locs: []*tikv.KeyLocation{
				kl("a", "m", 1), // Covers first range
				kl("t", "z", 2), // Covers second range - large gap is OK for discrete ranges
			},
			wantValid: true,
		},
		{
			name: "missing range coverage",
			ranges: []tikv.KeyRange{
				kr("a", "m"),
				kr("m", "z"),
			},
			locs:      []*tikv.KeyLocation{kl("a", "m", 1)},
			wantValid: false, // Invalid - second range not covered
		},

		// Edge cases
		{
			name:      "empty ranges with locations",
			ranges:    []tikv.KeyRange{},
			locs:      []*tikv.KeyLocation{kl("a", "z", 1)},
			wantValid: false, // Invalid - locations exist but no ranges to cover
		},
		{
			name:      "empty ranges without locations",
			ranges:    []tikv.KeyRange{},
			locs:      []*tikv.KeyLocation{},
			wantValid: true,
		},
		{
			name:      "empty locations",
			ranges:    []tikv.KeyRange{kr("a", "z")},
			locs:      []*tikv.KeyLocation{},
			wantValid: false,
		},
		{
			name: "exact boundary match",
			ranges: []tikv.KeyRange{
				kr("a", "m"),
				kr("m", "z"),
			},
			locs: []*tikv.KeyLocation{
				kl("a", "m", 1),
				kl("m", "z", 2),
			},
			wantValid: true,
		},
		{
			name:      "location boundary equals range start",
			ranges:    []tikv.KeyRange{kr("m", "z")},
			locs:      []*tikv.KeyLocation{kl("m", "z", 1)},
			wantValid: true,
		},

		// Monotonicity violations
		{
			name:   "locations not monotonic",
			ranges: []tikv.KeyRange{kr("a", "z")},
			locs: []*tikv.KeyLocation{
				kl("m", "z", 1),
				kl("a", "m", 2), // Out of order
			},
			wantValid: false,
		},
		{
			name:   "locations overlap",
			ranges: []tikv.KeyRange{kr("a", "z")},
			locs: []*tikv.KeyLocation{
				kl("a", "n", 1),
				kl("m", "z", 2), // Overlaps with previous location
			},
			wantValid: false,
		},
		{
			name:   "location extends to infinity and overlaps next - invalid",
			ranges: []tikv.KeyRange{kr("a", "z")},
			locs: []*tikv.KeyLocation{
				kl("a", "", 1),  // Extends to infinity
				kl("m", "z", 2), // Overlaps with previous (prev.EndKey=+inf > "m")
			},
			wantValid: false, // Invalid - locations overlap
		},

		// Unused location violations (Property 3)
		{
			name:   "extra location not covering any range",
			ranges: []tikv.KeyRange{kr("a", "b")},
			locs: []*tikv.KeyLocation{
				kl("a", "b", 1), // Covers the range
				kl("x", "z", 2), // Doesn't cover any range
			},
			wantValid: false, // Invalid - second location is unused
		},
		{
			name:   "middle location not covering any range",
			ranges: []tikv.KeyRange{kr("a", "c")},
			locs: []*tikv.KeyLocation{
				kl("a", "b", 1), // Covers part of range
				kl("b", "c", 2), // Covers rest of range
				kl("x", "z", 3), // Doesn't cover any range
			},
			wantValid: false, // Invalid - third location is unused
		},

		{
			name:   "current location starts from beginning after non-beginning",
			ranges: []tikv.KeyRange{kr("a", "z")},
			locs: []*tikv.KeyLocation{
				kl("a", "m", 1),
				kl("", "z", 2), // Starts from beginning after a non-beginning location
			},
			wantValid: false,
		},
		{
			name:   "valid: first location starts from beginning",
			ranges: []tikv.KeyRange{kr("", "z")},
			locs: []*tikv.KeyLocation{
				kl("", "m", 1), // First location can start from beginning
				kl("m", "z", 2),
			},
			wantValid: true,
		},
		{
			name:   "valid: last location extends to infinity",
			ranges: []tikv.KeyRange{kr("a", "")},
			locs: []*tikv.KeyLocation{
				kl("a", "m", 1),
				kl("m", "", 2), // Last location can extend to infinity
			},
			wantValid: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := validateLocationCoverage(ctx, tt.ranges, tt.locs)
			if got != tt.wantValid {
				t.Errorf("validateLocationCoverage() = %v, want %v", got, tt.wantValid)
			}
		})
	}
}

// TestPanicInSplitKeyRangesByBuckets tests that panic recovery correctly logs the location index
func TestPanicInSplitKeyRangesByBuckets(t *testing.T) {
	// Set up mock TiKV cluster with multiple regions
	mockClient, cluster, pdClient, err := testutils.NewMockTiKV("", nil)
	require.NoError(t, err)
	defer func() {
		pdClient.Close()
		err = mockClient.Close()
		require.NoError(t, err)
	}()

	// Create 4 regions: nil---g---n---x---nil
	_, regionIDs, _ := testutils.BootstrapWithMultiRegions(cluster, []byte("g"), []byte("n"), []byte("x"))
	// Add buckets to each region
	cluster.SplitRegionBuckets(regionIDs[0], [][]byte{{}, {'c'}, {'g'}}, regionIDs[0])
	cluster.SplitRegionBuckets(regionIDs[1], [][]byte{{'g'}, {'k'}, {'n'}}, regionIDs[1])
	cluster.SplitRegionBuckets(regionIDs[2], [][]byte{{'n'}, {'t'}, {'x'}}, regionIDs[2])
	cluster.SplitRegionBuckets(regionIDs[3], [][]byte{{'x'}, {}}, regionIDs[3])

	pdCli := tikv.NewCodecPDClient(tikv.ModeTxn, pdClient)
	defer pdCli.Close()

	cache := NewRegionCache(tikv.NewRegionCache(pdCli))
	defer cache.Close()

	bo := backoff.NewBackofferWithVars(context.Background(), 3000, nil)

	// Build key ranges that span multiple regions
	ranges := buildCopRanges("a", "z")

	// Enable the failpoint to trigger panic at location index 2 (the 3rd region)
	require.NoError(t, failpoint.Enable("github.com/pingcap/tidb/pkg/store/copr/panicInSplitKeyRangesByBuckets", "return(2)"))
	defer func() {
		require.NoError(t, failpoint.Disable("github.com/pingcap/tidb/pkg/store/copr/panicInSplitKeyRangesByBuckets"))
	}()

	// Call SplitKeyRangesByBuckets which should trigger the panic
	// The defer/recover in the function should catch it, log diagnostics, and re-panic
	didPanic := false
	panicValue := ""
	func() {
		defer func() {
			if r := recover(); r != nil {
				didPanic = true
				panicValue = r.(string)
				t.Logf("Successfully caught panic: %v", r)
			}
		}()

		// This will trigger the panic when processing location index 2
		_, err = cache.SplitKeyRangesByBuckets(bo, ranges)
		// Should not reach here
		t.Fatal("Expected panic but none occurred")
	}()

	// Verify that panic occurred and was re-panicked
	require.True(t, didPanic, "Expected panic to occur")
	require.Equal(t, "failpoint triggered panic in bucket splitting", panicValue)

	t.Logf("Test completed successfully - panic was caught, diagnostics logged, and re-panicked as expected")
}

// TestLocateBucketNilFallback checks unsplit fallback for out-of-region input
// and injected defensive bucket failures, including their bucket versions.
func TestLocateBucketNilFallback(t *testing.T) {
	ctx := context.Background()

	// Create a location covering [a, m)
	loc := &tikv.KeyLocation{
		Region:   tikv.NewRegionVerID(1, 0, 0),
		StartKey: []byte("a"),
		EndKey:   []byte("m"),
		Buckets: &metapb.Buckets{
			Keys:    [][]byte{[]byte("a"), []byte("f"), []byte("m")},
			Version: 1,
		},
	}

	// Create ranges where the first one is OUTSIDE the location (starts at "x").
	// This simulates the bug scenario where ranges don't match locations.
	outsideRanges := NewKeyRanges([]kv.KeyRange{
		{StartKey: []byte("x"), EndKey: []byte("z")}, // Outside [a, m)
	})

	// Create LocationKeyRanges with mismatched ranges
	lkr := &LocationKeyRanges{
		Location: loc,
		Ranges:   outsideRanges,
	}

	// Call splitKeyRangesByBuckets - should NOT panic, should return unsplit ranges
	result, fb := lkr.splitKeyRangesByBuckets(ctx)
	require.NotNil(t, fb)

	// Verify we got the fallback behavior: unsplit original LocationKeyRanges
	require.Len(t, result, 1, "Expected 1 unsplit LocationKeyRanges")
	require.Equal(t, lkr, result[0], "Expected original LocationKeyRanges to be returned")

	// Absent/empty buckets return region-only ranges even when the input starts
	// outside the location. They must not manufacture a version-zero fallback.
	for _, buckets := range []*metapb.Buckets{nil, {}, {Version: 7}} {
		withoutKeys := &LocationKeyRanges{
			Location: &tikv.KeyLocation{
				Region: loc.Region, StartKey: loc.StartKey, EndKey: loc.EndKey, Buckets: buckets,
			},
			Ranges: outsideRanges,
		}
		result, fallback := withoutKeys.splitKeyRangesByBuckets(ctx)
		require.Nil(t, fallback)
		require.Len(t, result, 1)
		require.Same(t, withoutKeys, result[0])
	}

	for _, version := range []uint64{0, 7} {
		versionName := "nonzero"
		if version == 0 {
			versionName = "zero"
		}
		for _, tc := range []struct {
			name   string
			reason string
		}{
			{name: "nil", reason: "locate_bucket_nil"},
			{name: "no_progress", reason: "bucket_not_contain_start_no_progress"},
		} {
			t.Run(tc.reason+"/"+versionName, func(t *testing.T) {
				inside := &LocationKeyRanges{
					Location: &tikv.KeyLocation{
						Region: loc.Region, StartKey: loc.StartKey, EndKey: loc.EndKey,
						Buckets: &metapb.Buckets{
							Keys: [][]byte{[]byte("a"), []byte("f"), []byte("m")}, Version: version,
						},
					},
					Ranges: buildCopRanges("b", "h"),
				}
				const hook = "github.com/pingcap/tidb/pkg/store/copr/bucketSplitFallbackForTest"
				require.NoError(t, failpoint.Enable(hook, "return(\""+tc.name+"\")"))
				defer func() { require.NoError(t, failpoint.Disable(hook)) }()
				result, fallback := inside.splitKeyRangesByBuckets(ctx)
				require.NotNil(t, fallback)
				require.Equal(t, tc.reason, fallback.reason)
				require.Equal(t, version, fallback.bucketVersion)
				require.Equal(t, []byte("b"), fallback.startKey)
				require.Equal(t, []byte("h"), fallback.endKey)
				require.Equal(t, 1, fallback.remainingRangeCount)
				require.Len(t, result, 1)
				require.Same(t, inside, result[0])
			})
		}
	}

	t.Log("StartKey outside location fallback working correctly - returned unsplit ranges instead of panicking")
}

// TestBucketFallbackWithoutSplitPreservesRegion checks that a range mismatch
// does not evict unchanged metadata or probe PD on every repeated fallback.
func TestBucketFallbackWithoutSplitPreservesRegion(t *testing.T) {
	t.Run("active_probe_and_completion_cooldown", func(t *testing.T) {
		cache := &RegionCache{}
		region := tikv.NewRegionVerID(1, 1, 1)
		started := time.Now()
		require.True(t, cache.allowBucketFallbackProbe(region, started))
		finished := started.Add(10 * bucketFallbackProbeInterval)
		require.False(t, cache.allowBucketFallbackProbe(region, finished),
			"an active probe must remain reserved beyond the cooldown interval")
		cache.finishBucketFallbackProbe(region, finished)
		require.False(t, cache.allowBucketFallbackProbe(region, finished))
		require.False(t, cache.allowBucketFallbackProbe(region, finished.Add(bucketFallbackProbeInterval-time.Nanosecond)))
		require.True(t, cache.allowBucketFallbackProbe(region, finished.Add(bucketFallbackProbeInterval)))
	})

	for _, tc := range []struct {
		name          string
		version       uint64
		changeConfVer bool
	}{
		{name: "zero", version: 0},
		{name: "nonzero", version: 7},
		{name: "zero_conf_ver_changed", version: 0, changeConfVer: true},
		{name: "nonzero_conf_ver_changed", version: 7, changeConfVer: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			testBucketFallbackWithoutSplitPreservesRegion(t, tc.version, tc.changeConfVer)
		})
	}
}

func testBucketFallbackWithoutSplitPreservesRegion(t *testing.T, bucketVersion uint64, changeConfVer bool) {
	t.Helper()
	mockClient, cluster, pdClient, err := testutils.NewMockTiKV("", nil)
	require.NoError(t, err)
	defer func() {
		require.NoError(t, mockClient.Close())
	}()
	_, regionIDs, _ := testutils.BootstrapWithMultiRegions(cluster, []byte("m"), []byte("z"))
	cluster.SplitRegionBuckets(regionIDs[1], [][]byte{[]byte("m"), []byte("t"), []byte("z")}, bucketVersion)
	pdCli := tikv.NewCodecPDClient(tikv.ModeTxn, pdClient)
	defer pdCli.Close()
	cache := NewRegionCache(tikv.NewRegionCache(pdCli))
	defer cache.Close()
	ctx := context.Background()
	bo := backoff.NewBackofferWithVars(ctx, 3000, nil)

	_, err = cache.SplitKeyRangesByLocations(bo, buildCopRanges("a", "z"), UnspecifiedLimit, false, true)
	require.NoError(t, err)
	cachedLoc := cache.TryLocateKey([]byte("m"))
	require.NotNil(t, cachedLoc)
	require.Equal(t, bucketVersion, cachedLoc.GetBucketVersion())
	cachedRegion := cache.GetCachedRegionWithRLock(cachedLoc.Region)
	require.NotNil(t, cachedRegion)

	if changeConfVer {
		// Adding a peer changes only the configuration epoch in PD.
		before, _ := cluster.GetRegion(regionIDs[1])
		storeID := cluster.AllocID()
		cluster.AddStore(storeID, "conf-ver-store")
		cluster.AddPeer(regionIDs[1], storeID, cluster.AllocID())
		after, _ := cluster.GetRegion(regionIDs[1])
		require.Greater(t, after.GetRegionEpoch().GetConfVer(), cachedLoc.Region.GetConfVer())
		require.Equal(t, cachedLoc.Region.GetVer(), after.GetRegionEpoch().GetVersion())
		require.Equal(t, before.GetStartKey(), after.GetStartKey())
		require.Equal(t, before.GetEndKey(), after.GetEndKey())
	}

	// Use the same mismatch as the split regression without changing PD's range.
	ranges := NewKeyRanges([]kv.KeyRange{
		{StartKey: []byte("a"), EndKey: []byte("z")},
		{StartKey: []byte("b"), EndKey: []byte("p")},
	})
	locs, err := cache.SplitKeyRangesByLocations(bo, ranges, UnspecifiedLimit, false, true)
	require.NoError(t, err)
	require.Len(t, locs, 2)
	_, fallback := locs[1].splitKeyRangesByBuckets(ctx)
	require.NotNil(t, fallback)
	require.Equal(t, "range_start_outside_location", fallback.reason)
	require.Equal(t, bucketVersion, fallback.bucketVersion)

	// Count entries into the direct PD probe path, not just returned epochs:
	// eviction followed by reloading unchanged metadata would keep the epoch.
	probes := 0
	const probeFailpoint = "github.com/pingcap/tidb/pkg/store/copr/beforeBucketFallbackPDProbe"
	require.NoError(t, failpoint.EnableCall(probeFailpoint, func() { probes++ }))
	defer func() {
		require.NoError(t, failpoint.Disable(probeFailpoint))
	}()
	result, err := cache.SplitKeyRangesByBuckets(bo, ranges)
	require.NoError(t, err)
	require.NotEmpty(t, result)
	require.Equal(t, 1, probes, "the first fallback should check PD")
	require.Same(t, cachedRegion, cache.GetCachedRegionWithRLock(cachedLoc.Region),
		"unchanged metadata must not be invalidated and reloaded")

	// Hold the reservation in the future so this assertion is independent of
	// test-machine speed. No sleeps or production clock changes are required.
	cache.bucketFallbackMu.Lock()
	cache.bucketFallbackNextProbe[cachedLoc.Region] = time.Now().Add(time.Hour)
	cache.bucketFallbackMu.Unlock()
	for range 3 {
		result, err = cache.SplitKeyRangesByBuckets(bo, ranges)
		require.NoError(t, err)
		require.NotEmpty(t, result)
		require.Equal(t, 1, probes, "repeated fallback must not probe PD again during the cooldown")
		require.Same(t, cachedRegion, cache.GetCachedRegionWithRLock(cachedLoc.Region))
		current := cache.TryLocateKey([]byte("m"))
		require.NotNil(t, current)
		require.Equal(t, cachedLoc.Region, current.Region)
		require.Equal(t, []byte("z"), current.EndKey)
	}

	// A negative check must not suppress recovery forever. After a real split,
	// expire the reservation deterministically and verify invalidation resumes.
	newRegionID, newPeerID := cluster.AllocID(), cluster.AllocID()
	newPeerIDs := []uint64{newPeerID}
	if changeConfVer {
		newPeerIDs = append(newPeerIDs, cluster.AllocID())
	}
	cluster.Split(regionIDs[1], newRegionID, []byte("t"), newPeerIDs, newPeerID)
	cache.bucketFallbackMu.Lock()
	cache.bucketFallbackNextProbe[cachedLoc.Region] = time.Now().Add(-bucketFallbackProbeInterval)
	cache.bucketFallbackMu.Unlock()
	_, err = cache.SplitKeyRangesByBuckets(bo, ranges)
	require.NoError(t, err)
	require.Equal(t, 2, probes)
	require.Nil(t, cache.GetCachedRegionWithRLock(cachedLoc.Region))
	right := cache.TryLocateKey([]byte("t"))
	require.NotNil(t, right)
	require.Equal(t, newRegionID, right.Region.GetID())
}

// TestLateBucketFallbackPreservesRefreshedRegions fixes the interleaving where
// one request holds pre-split locations while another refreshes both children.
func TestLateBucketFallbackPreservesRefreshedRegions(t *testing.T) {
	mockClient, cluster, pdClient, err := testutils.NewMockTiKV("", nil)
	require.NoError(t, err)
	defer func() {
		require.NoError(t, mockClient.Close())
	}()
	_, regionIDs, _ := testutils.BootstrapWithMultiRegions(cluster, []byte("m"), []byte("z"))
	const bucketVersion uint64 = 7
	cluster.SplitRegionBuckets(regionIDs[1], [][]byte{[]byte("m"), []byte("t"), []byte("z")}, bucketVersion)
	pdCli := tikv.NewCodecPDClient(tikv.ModeTxn, pdClient)
	defer pdCli.Close()
	cache := NewRegionCache(tikv.NewRegionCache(pdCli))
	defer cache.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	warmBO := backoff.NewBackofferWithVars(ctx, 3000, nil)
	_, err = cache.SplitKeyRangesByLocations(warmBO, buildCopRanges("a", "z"), UnspecifiedLimit, false, true)
	require.NoError(t, err)
	staleLoc := cache.TryLocateKey([]byte("m"))
	require.NotNil(t, staleLoc)
	require.Equal(t, bucketVersion, staleLoc.GetBucketVersion())
	require.Equal(t, []byte("z"), staleLoc.EndKey)

	newRegionID, newPeerID := cluster.AllocID(), cluster.AllocID()
	cluster.Split(regionIDs[1], newRegionID, []byte("t"), []uint64{newPeerID}, newPeerID)

	// Request A has already detected fallback and captured its old location
	// when it reaches this barrier. Request B will refresh while A is paused.
	entered := make(chan struct{}, 1)
	resume := make(chan struct{})
	released := false
	release := func() {
		if !released {
			close(resume)
			released = true
		}
	}
	const probeFailpoint = "github.com/pingcap/tidb/pkg/store/copr/beforeBucketFallbackPDProbe"
	require.NoError(t, failpoint.EnableCall(probeFailpoint, func() {
		select {
		case entered <- struct{}{}:
		default:
		}
		select {
		case <-resume:
		case <-ctx.Done():
		}
	}))
	defer func() {
		require.NoError(t, failpoint.Disable(probeFailpoint))
	}()

	type fallbackResult struct {
		locs []*LocationKeyRanges
		err  error
	}
	results := make(chan fallbackResult, 1)
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		lateBO := backoff.NewBackofferWithVars(ctx, 3000, nil)
		ranges := NewKeyRanges([]kv.KeyRange{
			{StartKey: []byte("a"), EndKey: []byte("z")},
			{StartKey: []byte("b"), EndKey: []byte("p")},
		})
		locs, err := cache.SplitKeyRangesByBuckets(lateBO, ranges)
		results <- fallbackResult{locs: locs, err: err}
	}()
	// Release and join A before disabling the hook or closing the cache,
	// including when a main-goroutine assertion fails.
	defer func() {
		cancel()
		release()
		select {
		case <-exited:
		case <-time.After(5 * time.Second):
			t.Error("late fallback did not stop after cancellation")
		}
	}()
	select {
	case <-entered:
	case result := <-results:
		t.Fatalf("fallback did not reach the barrier (enable failpoints): %v", result.err)
	case <-ctx.Done():
		t.Fatal("timed out waiting for the late fallback barrier")
	}
	// A is paused inside its probe. Advancing the admission clock must not
	// admit another probe for this epoch, even after the old cooldown elapsed.
	require.False(t, cache.allowBucketFallbackProbe(staleLoc.Region, time.Now().Add(10*bucketFallbackProbeInterval)))
	require.NotNil(t, cache.GetCachedRegionWithRLock(staleLoc.Region))

	// Request B performs the ordinary error-driven invalidate-and-reload path.
	// Its backoffer is separate from A's; the test does not share mutable request
	// state between goroutines.
	refreshBO := backoff.NewBackofferWithVars(ctx, 3000, nil)
	cache.InvalidateCachedRegion(staleLoc.Region)
	refreshed, err := cache.SplitKeyRangesByLocations(refreshBO, buildCopRanges("m", "z"), UnspecifiedLimit, false, true)
	require.NoError(t, err)
	require.Len(t, refreshed, 2)
	leftLoc, rightLoc := refreshed[0].Location, refreshed[1].Location
	require.Equal(t, regionIDs[1], leftLoc.Region.GetID())
	require.Greater(t, leftLoc.Region.GetVer(), staleLoc.Region.GetVer())
	require.Equal(t, []byte("t"), leftLoc.EndKey)
	require.Equal(t, newRegionID, rightLoc.Region.GetID())
	require.Equal(t, []byte("t"), rightLoc.StartKey)
	require.Equal(t, []byte("z"), rightLoc.EndKey)
	leftRegion := cache.GetCachedRegionWithRLock(leftLoc.Region)
	rightRegion := cache.GetCachedRegionWithRLock(rightLoc.Region)
	require.NotNil(t, leftRegion)
	require.NotNil(t, rightRegion)
	require.Nil(t, cache.GetCachedRegionWithRLock(staleLoc.Region))

	// A now sees PD's newer boundaries but must not invalidate B's entries.
	release()
	var result fallbackResult
	select {
	case result = <-results:
	case <-ctx.Done():
		t.Fatal("timed out waiting for the late fallback to finish")
	}
	require.NoError(t, result.err)
	require.NotEmpty(t, result.locs)
	for _, loc := range result.locs {
		require.NotEqual(t, staleLoc.Region, loc.Location.Region,
			"the late request must rebuild with current region epochs")
	}
	require.Same(t, leftRegion, cache.GetCachedRegionWithRLock(leftLoc.Region),
		"late fallback must not evict and reload the refreshed left child")
	require.Same(t, rightRegion, cache.GetCachedRegionWithRLock(rightLoc.Region),
		"late fallback must preserve the refreshed right child")
	require.Nil(t, cache.GetCachedRegionWithRLock(staleLoc.Region))
}

// TestLateZeroVersionBucketFallbackPreservesRefreshedBuckets verifies that
// zero is an unknown bucket version, not a wildcard matching a later generation.
func TestLateZeroVersionBucketFallbackPreservesRefreshedBuckets(t *testing.T) {
	mockClient, cluster, pdClient, err := testutils.NewMockTiKV("", nil)
	require.NoError(t, err)
	defer func() {
		require.NoError(t, mockClient.Close())
	}()
	_, regionIDs, _ := testutils.BootstrapWithMultiRegions(cluster, []byte("m"), []byte("z"))
	const bucketVersion uint64 = 0
	cluster.SplitRegionBuckets(regionIDs[1], [][]byte{[]byte("m"), []byte("t"), []byte("z")}, bucketVersion)
	pdCli := tikv.NewCodecPDClient(tikv.ModeTxn, pdClient)
	defer pdCli.Close()
	cache := NewRegionCache(tikv.NewRegionCache(pdCli))
	defer cache.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	warmBO := backoff.NewBackofferWithVars(ctx, 3000, nil)
	_, err = cache.SplitKeyRangesByLocations(warmBO, buildCopRanges("a", "z"), UnspecifiedLimit, false, true)
	require.NoError(t, err)
	staleLoc := cache.TryLocateKey([]byte("m"))
	require.NotNil(t, staleLoc)
	require.Equal(t, bucketVersion, staleLoc.GetBucketVersion())
	require.Equal(t, []byte("z"), staleLoc.EndKey)

	// Request A captures version-zero buckets before this barrier. Request B
	// will replace them without changing the region epoch while A is paused.
	entered := make(chan struct{}, 1)
	resume := make(chan struct{})
	released := false
	release := func() {
		if !released {
			close(resume)
			released = true
		}
	}
	const probeFailpoint = "github.com/pingcap/tidb/pkg/store/copr/beforeBucketFallbackPDProbe"
	require.NoError(t, failpoint.EnableCall(probeFailpoint, func() {
		select {
		case entered <- struct{}{}:
		default:
		}
		select {
		case <-resume:
		case <-ctx.Done():
		}
	}))
	defer func() {
		require.NoError(t, failpoint.Disable(probeFailpoint))
	}()

	type fallbackResult struct {
		locs []*LocationKeyRanges
		err  error
	}
	results := make(chan fallbackResult, 1)
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		lateBO := backoff.NewBackofferWithVars(ctx, 3000, nil)
		ranges := NewKeyRanges([]kv.KeyRange{
			{StartKey: []byte("a"), EndKey: []byte("z")},
			{StartKey: []byte("b"), EndKey: []byte("p")},
		})
		locs, err := cache.SplitKeyRangesByBuckets(lateBO, ranges)
		results <- fallbackResult{locs: locs, err: err}
	}()
	// Release and join A before disabling the hook or closing the cache,
	// including when a main-goroutine assertion fails.
	defer func() {
		cancel()
		release()
		select {
		case <-exited:
		case <-time.After(5 * time.Second):
			t.Error("late fallback did not stop after cancellation")
		}
	}()
	select {
	case <-entered:
	case result := <-results:
		t.Fatalf("fallback did not reach the barrier (enable failpoints): %v", result.err)
	case <-ctx.Done():
		t.Fatal("timed out waiting for the late fallback barrier")
	}
	// A is paused inside its probe. Advancing the admission clock must not
	// admit another probe for this epoch, even after the old cooldown elapsed.
	require.False(t, cache.allowBucketFallbackProbe(staleLoc.Region, time.Now().Add(10*bucketFallbackProbeInterval)))
	require.NotNil(t, cache.GetCachedRegionWithRLock(staleLoc.Region))

	// Request B refreshes only the buckets; the region epoch stays the same.
	const refreshedBucketVersion uint64 = 8
	cluster.SplitRegionBuckets(regionIDs[1], [][]byte{[]byte("m"), []byte("t"), []byte("z")}, refreshedBucketVersion)
	refreshBO := backoff.NewBackofferWithVars(ctx, 3000, nil)
	cache.InvalidateCachedRegion(staleLoc.Region)
	refreshed, err := cache.SplitKeyRangesByLocations(refreshBO, buildCopRanges("m", "z"), UnspecifiedLimit, false, true)
	require.NoError(t, err)
	require.Len(t, refreshed, 1)
	refreshedLoc := refreshed[0].Location
	require.Equal(t, staleLoc.Region, refreshedLoc.Region)
	require.Equal(t, refreshedBucketVersion, refreshedLoc.GetBucketVersion())
	refreshedRegion := cache.GetCachedRegionWithRLock(refreshedLoc.Region)
	require.NotNil(t, refreshedRegion)

	// Now split in PD without refreshing this cache again. A's probe will see a
	// real range change, so only the bucket-version guard can prevent A from
	// evicting B's replacement under the same RegionVerID. A later request using
	// version 8 can perform its own recovery after the probe cooldown.
	newRegionID, newPeerID := cluster.AllocID(), cluster.AllocID()
	cluster.Split(regionIDs[1], newRegionID, []byte("t"), []uint64{newPeerID}, newPeerID)

	// A sees PD's newer boundaries but must preserve B's bucket generation.
	release()
	var result fallbackResult
	select {
	case result = <-results:
	case <-ctx.Done():
		t.Fatal("timed out waiting for the late fallback to finish")
	}
	require.NoError(t, result.err)
	require.NotEmpty(t, result.locs)
	require.Same(t, refreshedRegion, cache.GetCachedRegionWithRLock(refreshedLoc.Region),
		"late version-zero fallback must not evict the nonzero bucket generation")
	current := cache.TryLocateKey([]byte("m"))
	require.NotNil(t, current)
	require.Equal(t, refreshedLoc.Region, current.Region)
	require.Equal(t, refreshedBucketVersion, current.GetBucketVersion())
}

// TestStaleBucketFallbackInvalidatesRegion checks that fallback evicts the
// cached pre-split epoch before rebuilding tasks without buckets.
func TestStaleBucketFallbackInvalidatesRegion(t *testing.T) {
	for _, tc := range []struct {
		name          string
		version       uint64
		changeConfVer bool
	}{
		{name: "zero", version: 0},
		{name: "nonzero", version: 7},
		{name: "zero_conf_ver_changed", version: 0, changeConfVer: true},
		{name: "nonzero_conf_ver_changed", version: 7, changeConfVer: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			testStaleBucketFallbackInvalidatesRegion(t, tc.version, tc.changeConfVer)
		})
	}
}

func testStaleBucketFallbackInvalidatesRegion(t *testing.T, bucketVersion uint64, changeConfVer bool) {
	t.Helper()
	mockClient, cluster, pdClient, err := testutils.NewMockTiKV("", nil)
	require.NoError(t, err)
	defer func() {
		require.NoError(t, mockClient.Close())
	}()

	_, regionIDs, _ := testutils.BootstrapWithMultiRegions(cluster, []byte("m"), []byte("z"))
	cluster.SplitRegionBuckets(regionIDs[1], [][]byte{[]byte("m"), []byte("t"), []byte("z")}, bucketVersion)

	pdCli := tikv.NewCodecPDClient(tikv.ModeTxn, pdClient)
	defer pdCli.Close()
	cache := NewRegionCache(tikv.NewRegionCache(pdCli))
	defer cache.Close()
	ctx := context.Background()
	bo := backoff.NewBackofferWithVars(ctx, 3000, nil)

	// Warm both regions, including buckets, before PD learns about the split.
	_, err = cache.SplitKeyRangesByLocations(bo, buildCopRanges("a", "z"), UnspecifiedLimit, false, true)
	require.NoError(t, err)
	staleLoc := cache.TryLocateKey([]byte("m"))
	require.NotNil(t, staleLoc)
	require.Equal(t, regionIDs[1], staleLoc.Region.GetID())
	require.Equal(t, bucketVersion, staleLoc.GetBucketVersion())
	require.Equal(t, []byte("z"), staleLoc.EndKey)
	require.NotNil(t, cache.GetCachedRegionWithRLock(staleLoc.Region))

	newRegionID, newPeerID := cluster.AllocID(), cluster.AllocID()
	cluster.Split(regionIDs[1], newRegionID, []byte("t"), []uint64{newPeerID}, newPeerID)
	if changeConfVer {
		// The retained left region has both a new range and configuration epoch.
		storeID := cluster.AllocID()
		cluster.AddStore(storeID, "conf-ver-store")
		cluster.AddPeer(regionIDs[1], storeID, cluster.AllocID())
	}
	pdRegion, _ := cluster.GetRegion(regionIDs[1])
	require.Greater(t, pdRegion.GetRegionEpoch().GetVersion(), staleLoc.Region.GetVer())
	if changeConfVer {
		require.Greater(t, pdRegion.GetRegionEpoch().GetConfVer(), staleLoc.Region.GetConfVer())
	}
	stillCached := cache.TryLocateKey([]byte("m"))
	require.NotNil(t, stillCached)
	require.Equal(t, staleLoc.Region, stillCached.Region)
	require.Equal(t, []byte("z"), stillCached.EndKey)

	// The contained range becomes out of order after the region/bucket splits:
	// [a,z), [b,p) -> [m,z), [b,p) -> [t,z), [b,p).
	// This deterministically reaches the outside-location fallback with the
	// bucket version from the still-cached pre-split descriptor.
	ranges := NewKeyRanges([]kv.KeyRange{
		{StartKey: []byte("a"), EndKey: []byte("z")},
		{StartKey: []byte("b"), EndKey: []byte("p")},
	})
	locs, err := cache.SplitKeyRangesByLocations(bo, ranges, UnspecifiedLimit, false, true)
	require.NoError(t, err)
	require.Len(t, locs, 2)
	require.Equal(t, staleLoc.Region, locs[1].Location.Region)
	_, fallback := locs[1].splitKeyRangesByBuckets(ctx)
	require.NotNil(t, fallback)
	require.Equal(t, "range_start_outside_location", fallback.reason)
	require.Equal(t, bucketVersion, fallback.bucketVersion)
	require.NotNil(t, cache.GetCachedRegionWithRLock(staleLoc.Region),
		"detecting fallback alone must not invalidate the shared cache")

	// Exercise the outer fallback handler; do not invalidate the cache in the
	// test. Without the fix, the bucket-less rebuild reuses the old epoch.
	result, err := cache.SplitKeyRangesByBuckets(bo, ranges)
	require.NoError(t, err)
	require.NotEmpty(t, result)
	require.Nil(t, cache.GetCachedRegionWithRLock(staleLoc.Region),
		"fallback must evict the pre-split region epoch")

	left := cache.TryLocateKey([]byte("m"))
	require.NotNil(t, left)
	require.Equal(t, regionIDs[1], left.Region.GetID())
	require.Greater(t, left.Region.GetVer(), staleLoc.Region.GetVer())
	require.Equal(t, pdRegion.GetRegionEpoch().GetConfVer(), left.Region.GetConfVer())
	require.Equal(t, []byte("t"), left.EndKey)
	right := cache.TryLocateKey([]byte("t"))
	require.NotNil(t, right)
	require.Equal(t, newRegionID, right.Region.GetID())
	require.Equal(t, []byte("t"), right.StartKey)
	require.Equal(t, []byte("z"), right.EndKey)
}

// TestLocateBucketOutsideRegionNonNilFallback tests a subtle stale-bucket case:
// LocateBucket can return a non-nil bucket even when key is outside the region boundaries.
// Our bucket splitting must not livelock and should fall back safely.
func TestLocateBucketOutsideRegionNonNilFallback(t *testing.T) {
	ctx := context.Background()

	// Location covers [m, z). Buckets are stale (start before region start), so LocateBucket("b")
	// returns a non-nil bucket, clamping falls back to region boundaries, and bucket.Contains("b") is false.
	loc := &tikv.KeyLocation{
		Region:   tikv.NewRegionVerID(1, 0, 0),
		StartKey: []byte("m"),
		EndKey:   []byte("z"),
		Buckets: &metapb.Buckets{
			Keys:    [][]byte{[]byte("a"), []byte("f"), []byte("z")},
			Version: 1,
		},
	}

	startKey := []byte("b") // outside [m, z)
	require.False(t, loc.Contains(startKey), "sanity: startKey should be outside location")
	b := loc.LocateBucket(startKey)
	require.NotNil(t, b, "LocateBucket should return a non-nil bucket for this stale metadata")
	require.False(t, b.Contains(startKey), "sanity: clamped bucket should not contain the key")

	lkr := &LocationKeyRanges{
		Location: loc,
		Ranges:   NewKeyRanges([]kv.KeyRange{{StartKey: startKey, EndKey: []byte("c")}}),
	}

	result, fb := lkr.splitKeyRangesByBuckets(ctx)
	require.NotNil(t, fb)
	require.Len(t, result, 1, "Expected unsplit LocationKeyRanges fallback")
	require.Equal(t, lkr, result[0], "Expected original LocationKeyRanges to be returned")
}

func TestOverlappingRangesCanProduceOutOfOrderRangesAfterSplit(t *testing.T) {
	ctx := context.Background()

	// This test demonstrates a concrete root-cause for `range.StartKey < location.StartKey`:
	//
	// 1) Input key ranges are overlapping/contained (e.g. [a,z) contains [b,c)).
	// 2) `splitKeyRangesByLocation` splits the first range at the location boundary and stores the remainder
	//    as `ranges.first`, but keeps the still-unprocessed contained range in `ranges.mid`.
	// 3) The remaining `KeyRanges` becomes non-monotonic: [m,z) then [b,c).
	// 4) A later location can end up receiving a range whose start is < location.StartKey, which triggers
	//    the bucket-splitting fallback guard (and used to be able to livelock).

	loc0 := &tikv.KeyLocation{
		Region:   tikv.NewRegionVerID(1, 0, 0),
		StartKey: []byte("a"),
		EndKey:   []byte("m"),
	}
	loc1 := &tikv.KeyLocation{
		Region:   tikv.NewRegionVerID(2, 0, 0),
		StartKey: []byte("m"),
		EndKey:   []byte("z"),
		Buckets: &metapb.Buckets{
			// One bucket covering the full location is enough to exercise the
			// bucket-splitting loop and reach the mismatch.
			Keys:    [][]byte{[]byte("m"), []byte("z")},
			Version: 1,
		},
	}

	ranges := NewKeyRanges([]kv.KeyRange{
		{StartKey: []byte("a"), EndKey: []byte("z")}, // spans loc0 -> loc1
		{StartKey: []byte("b"), EndKey: []byte("c")}, // fully inside loc0, contained by the first range
	})

	cache := &RegionCache{}
	res := []*LocationKeyRanges{}
	_, remaining, isBreak := cache.splitKeyRangesByLocation(ctx, loc0, ranges, res)
	require.False(t, isBreak, "should not break: more locations remain")
	require.Equal(t, 2, remaining.Len(), "expect [m,z) + original contained range to remain")

	require.Equal(t, "m", string(remaining.At(0).StartKey))
	require.Equal(t, "z", string(remaining.At(0).EndKey))
	require.Equal(t, "b", string(remaining.At(1).StartKey))
	require.Equal(t, "c", string(remaining.At(1).EndKey))

	// Now bucket splitting must observe that remaining ranges are inconsistent with loc1 boundaries
	// and fall back safely instead of looping forever.
	lkr := &LocationKeyRanges{Location: loc1, Ranges: remaining}
	result, fb := lkr.splitKeyRangesByBuckets(ctx)
	require.NotNil(t, fb)
	require.Equal(t, "range_start_outside_location", fb.reason)
	require.Len(t, result, 1)
	require.Equal(t, lkr, result[0])
}
