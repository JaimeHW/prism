//! Exhaustive tests for the PGO profile data system.

#[cfg(test)]
mod tests {
    use crate::runtime::profile_data::*;

    // =========================================================================
    // BranchProfile Tests
    // =========================================================================

    #[test]
    fn test_branch_profile_new() {
        let bp = BranchProfile::new();
        assert_eq!(bp.taken, 0);
        assert_eq!(bp.not_taken, 0);
        assert_eq!(bp.total(), 0);
    }

    #[test]
    fn test_branch_profile_with_counts() {
        let bp = BranchProfile::with_counts(100, 50);
        assert_eq!(bp.taken, 100);
        assert_eq!(bp.not_taken, 50);
        assert_eq!(bp.total(), 150);
    }

    #[test]
    fn test_branch_profile_default() {
        let bp = BranchProfile::default();
        assert_eq!(bp.taken, 0);
        assert_eq!(bp.not_taken, 0);
    }

    #[test]
    fn test_branch_taken_probability_no_data() {
        let bp = BranchProfile::new();
        assert!((bp.taken_probability() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_branch_taken_probability_all_taken() {
        let bp = BranchProfile::with_counts(100, 0);
        assert!((bp.taken_probability() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_branch_taken_probability_all_not_taken() {
        let bp = BranchProfile::with_counts(0, 100);
        assert!((bp.taken_probability() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_branch_taken_probability_even_split() {
        let bp = BranchProfile::with_counts(50, 50);
        assert!((bp.taken_probability() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_branch_taken_probability_biased() {
        let bp = BranchProfile::with_counts(95, 5);
        assert!((bp.taken_probability() - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_branch_not_taken_probability() {
        let bp = BranchProfile::with_counts(30, 70);
        assert!((bp.not_taken_probability() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_branch_is_biased_highly_taken() {
        let bp = BranchProfile::with_counts(95, 5);
        assert!(bp.is_biased());
    }

    #[test]
    fn test_branch_is_biased_highly_not_taken() {
        let bp = BranchProfile::with_counts(5, 95);
        assert!(bp.is_biased());
    }

    #[test]
    fn test_branch_is_not_biased() {
        let bp = BranchProfile::with_counts(60, 40);
        assert!(!bp.is_biased());
    }

    #[test]
    fn test_branch_is_cold() {
        let bp = BranchProfile::with_counts(0, 200);
        assert!(bp.is_cold());
    }

    #[test]
    fn test_branch_is_not_cold_insufficient_data() {
        let bp = BranchProfile::with_counts(0, 50);
        assert!(!bp.is_cold());
    }

    #[test]
    fn test_branch_is_not_cold_warm() {
        let bp = BranchProfile::with_counts(10, 200);
        assert!(!bp.is_cold());
    }

    #[test]
    fn test_branch_merge() {
        let mut bp1 = BranchProfile::with_counts(10, 20);
        let bp2 = BranchProfile::with_counts(30, 40);
        bp1.merge(&bp2);
        assert_eq!(bp1.taken, 40);
        assert_eq!(bp1.not_taken, 60);
    }

    #[test]
    fn test_branch_merge_saturation() {
        let mut bp1 = BranchProfile::with_counts(u64::MAX - 5, 0);
        let bp2 = BranchProfile::with_counts(10, 0);
        bp1.merge(&bp2);
        assert_eq!(bp1.taken, u64::MAX);
    }

    #[test]
    fn test_branch_scale() {
        let mut bp = BranchProfile::with_counts(100, 200);
        bp.scale(0.5);
        assert_eq!(bp.taken, 50);
        assert_eq!(bp.not_taken, 100);
    }

    #[test]
    fn test_branch_scale_zero() {
        let mut bp = BranchProfile::with_counts(100, 200);
        bp.scale(0.0);
        assert_eq!(bp.taken, 0);
        assert_eq!(bp.not_taken, 0);
    }

    // =========================================================================
    // TypeProfile Tests
    // =========================================================================

    #[test]
    fn test_type_profile_new() {
        let tp = TypeProfile::new();
        assert_eq!(tp.total(), 0);
        assert!(tp.entries().is_empty());
    }

    #[test]
    fn test_type_profile_default() {
        let tp = TypeProfile::default();
        assert_eq!(tp.total(), 0);
    }

    #[test]
    fn test_type_profile_record_single() {
        let mut tp = TypeProfile::new();
        tp.record(1); // IntInt
        assert_eq!(tp.total(), 1);
        assert_eq!(tp.entries().len(), 1);
        assert_eq!(tp.entries()[0].type_id, 1);
        assert_eq!(tp.entries()[0].count, 1);
    }

    #[test]
    fn test_type_profile_record_repeated() {
        let mut tp = TypeProfile::new();
        for _ in 0..50 {
            tp.record(1); // IntInt
        }
        assert_eq!(tp.total(), 50);
        assert_eq!(tp.entries().len(), 1);
        assert_eq!(tp.entries()[0].count, 50);
    }

    #[test]
    fn test_type_profile_record_multiple_types() {
        let mut tp = TypeProfile::new();
        for _ in 0..30 {
            tp.record(1);
        }
        for _ in 0..20 {
            tp.record(2);
        }
        assert_eq!(tp.total(), 50);
        assert_eq!(tp.entries().len(), 2);
    }

    #[test]
    fn test_type_profile_dominant_type() {
        let mut tp = TypeProfile::new();
        for _ in 0..90 {
            tp.record(1);
        }
        for _ in 0..10 {
            tp.record(2);
        }
        let dom = tp.dominant_type().unwrap();
        assert_eq!(dom.type_id, 1);
        assert_eq!(dom.count, 90);
    }

    #[test]
    fn test_type_profile_dominant_type_empty() {
        let tp = TypeProfile::new();
        assert!(tp.dominant_type().is_none());
    }

    #[test]
    fn test_type_profile_is_monomorphic() {
        let mut tp = TypeProfile::new();
        for _ in 0..100 {
            tp.record(1);
        }
        assert!(tp.is_monomorphic());
    }

    #[test]
    fn test_type_profile_not_monomorphic_insufficient_data() {
        let mut tp = TypeProfile::new();
        for _ in 0..5 {
            tp.record(1);
        }
        assert!(!tp.is_monomorphic());
    }

    #[test]
    fn test_type_profile_not_monomorphic_mixed() {
        let mut tp = TypeProfile::new();
        for _ in 0..60 {
            tp.record(1);
        }
        for _ in 0..40 {
            tp.record(2);
        }
        assert!(!tp.is_monomorphic());
    }

    #[test]
    fn test_type_profile_is_polymorphic() {
        let mut tp = TypeProfile::new();
        for _ in 0..40 {
            tp.record(1);
        }
        for _ in 0..30 {
            tp.record(2);
        }
        for _ in 0..30 {
            tp.record(3);
        }
        assert!(tp.is_polymorphic());
    }

    #[test]
    fn test_type_profile_is_megamorphic() {
        let mut tp = TypeProfile::new();
        for i in 0..6u8 {
            for _ in 0..20 {
                tp.record(i);
            }
        }
        assert!(tp.is_megamorphic());
    }

    #[test]
    fn test_type_profile_merge() {
        let mut tp1 = TypeProfile::new();
        tp1.record(1);
        tp1.record(1);

        let mut tp2 = TypeProfile::new();
        tp2.record(1);
        tp2.record(2);

        tp1.merge(&tp2);
        assert_eq!(tp1.total(), 4);
        assert_eq!(tp1.entries().len(), 2);
    }

    // =========================================================================
    // CallProfile Tests
    // =========================================================================

    #[test]
    fn test_call_profile_new() {
        let cp = CallProfile::new();
        assert_eq!(cp.total(), 0);
        assert!(cp.targets().is_empty());
    }

    #[test]
    fn test_call_profile_default() {
        let cp = CallProfile::default();
        assert_eq!(cp.total(), 0);
    }

    #[test]
    fn test_call_profile_record_single() {
        let mut cp = CallProfile::new();
        cp.record(42);
        assert_eq!(cp.total(), 1);
        assert_eq!(cp.target_count(), 1);
        assert_eq!(cp.targets()[0].target_id, 42);
    }

    #[test]
    fn test_call_profile_record_multiple_same() {
        let mut cp = CallProfile::new();
        for _ in 0..100 {
            cp.record(42);
        }
        assert_eq!(cp.total(), 100);
        assert_eq!(cp.target_count(), 1);
        assert_eq!(cp.targets()[0].count, 100);
    }

    #[test]
    fn test_call_profile_record_multiple_targets() {
        let mut cp = CallProfile::new();
        for _ in 0..70 {
            cp.record(1);
        }
        for _ in 0..30 {
            cp.record(2);
        }
        assert_eq!(cp.total(), 100);
        assert_eq!(cp.target_count(), 2);
        // Should be sorted by frequency
        assert_eq!(cp.targets()[0].target_id, 1);
        assert_eq!(cp.targets()[0].count, 70);
    }

    #[test]
    fn test_call_profile_primary_target() {
        let mut cp = CallProfile::new();
        cp.record(10);
        cp.record(10);
        cp.record(20);
        let primary = cp.primary_target().unwrap();
        assert_eq!(primary.target_id, 10);
        assert_eq!(primary.count, 2);
    }

    #[test]
    fn test_call_profile_primary_target_empty() {
        let cp = CallProfile::new();
        assert!(cp.primary_target().is_none());
    }

    #[test]
    fn test_call_profile_is_monomorphic() {
        let mut cp = CallProfile::new();
        for _ in 0..100 {
            cp.record(42);
        }
        assert!(cp.is_monomorphic());
    }

    #[test]
    fn test_call_profile_not_monomorphic_mixed() {
        let mut cp = CallProfile::new();
        for _ in 0..50 {
            cp.record(1);
        }
        for _ in 0..50 {
            cp.record(2);
        }
        assert!(!cp.is_monomorphic());
    }

    #[test]
    fn test_call_profile_not_monomorphic_insufficient() {
        let mut cp = CallProfile::new();
        for _ in 0..5 {
            cp.record(1);
        }
        assert!(!cp.is_monomorphic());
    }

    #[test]
    fn test_call_profile_merge() {
        let mut cp1 = CallProfile::new();
        cp1.record(1);
        cp1.record(1);

        let mut cp2 = CallProfile::new();
        cp2.record(1);
        cp2.record(2);

        cp1.merge(&cp2);
        assert_eq!(cp1.total(), 4);
        assert_eq!(cp1.target_count(), 2);
        // Target 1 should have count 3
        assert_eq!(cp1.targets()[0].target_id, 1);
        assert_eq!(cp1.targets()[0].count, 3);
    }

    // =========================================================================
    // AtomicBranchCounter Tests
    // =========================================================================

    #[test]
    fn test_atomic_counter_new() {
        let counter = AtomicBranchCounter::new();
        let snapshot = counter.snapshot();
        assert_eq!(snapshot.taken, 0);
        assert_eq!(snapshot.not_taken, 0);
    }

    #[test]
    fn test_atomic_counter_default() {
        let counter = AtomicBranchCounter::default();
        let snapshot = counter.snapshot();
        assert_eq!(snapshot.total(), 0);
    }

    #[test]
    fn test_atomic_counter_record_taken() {
        let counter = AtomicBranchCounter::new();
        counter.record_taken();
        counter.record_taken();
        counter.record_taken();
        let snapshot = counter.snapshot();
        assert_eq!(snapshot.taken, 3);
        assert_eq!(snapshot.not_taken, 0);
    }

    #[test]
    fn test_atomic_counter_record_not_taken() {
        let counter = AtomicBranchCounter::new();
        counter.record_not_taken();
        counter.record_not_taken();
        let snapshot = counter.snapshot();
        assert_eq!(snapshot.taken, 0);
        assert_eq!(snapshot.not_taken, 2);
    }

    #[test]
    fn test_atomic_counter_mixed() {
        let counter = AtomicBranchCounter::new();
        counter.record_taken();
        counter.record_not_taken();
        counter.record_taken();
        let snapshot = counter.snapshot();
        assert_eq!(snapshot.taken, 2);
        assert_eq!(snapshot.not_taken, 1);
    }

    #[test]
    fn test_atomic_counter_reset() {
        let counter = AtomicBranchCounter::new();
        counter.record_taken();
        counter.record_taken();
        counter.record_not_taken();
        counter.reset();
        let snapshot = counter.snapshot();
        assert_eq!(snapshot.taken, 0);
        assert_eq!(snapshot.not_taken, 0);
    }

    #[test]
    fn test_atomic_counter_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let counter = Arc::new(AtomicBranchCounter::new());
        let mut handles = vec![];

        for _ in 0..4 {
            let c = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    c.record_taken();
                    c.record_not_taken();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let snapshot = counter.snapshot();
        assert_eq!(snapshot.taken, 4000);
        assert_eq!(snapshot.not_taken, 4000);
    }

    // =========================================================================
    // ProfileData Tests
    // =========================================================================

    #[test]
    fn test_profile_data_new() {
        let pd = ProfileData::new(42);
        assert_eq!(pd.code_id(), 42);
        assert_eq!(pd.execution_count(), 0);
        assert_eq!(pd.branch_count(), 0);
        assert_eq!(pd.type_count(), 0);
        assert_eq!(pd.call_count(), 0);
    }

    #[test]
    fn test_profile_data_record_execution() {
        let mut pd = ProfileData::new(1);
        pd.record_execution();
        pd.record_execution();
        assert_eq!(pd.execution_count(), 2);
    }

    #[test]
    fn test_profile_data_is_hot() {
        let mut pd = ProfileData::new(1);
        for _ in 0..100 {
            pd.record_execution();
        }
        assert!(pd.is_hot(50));
        assert!(pd.is_hot(100));
        assert!(!pd.is_hot(101));
    }

    #[test]
    fn test_profile_data_record_branch() {
        let mut pd = ProfileData::new(1);
        pd.record_branch(10, true);
        pd.record_branch(10, true);
        pd.record_branch(10, false);
        let bp = pd.branch_at(10).unwrap();
        assert_eq!(bp.taken, 2);
        assert_eq!(bp.not_taken, 1);
    }

    #[test]
    fn test_profile_data_branch_at_missing() {
        let pd = ProfileData::new(1);
        assert!(pd.branch_at(99).is_none());
    }

    #[test]
    fn test_profile_data_record_type() {
        let mut pd = ProfileData::new(1);
        pd.record_type(20, 1);
        pd.record_type(20, 1);
        pd.record_type(20, 2);
        let tp = pd.type_at(20).unwrap();
        assert_eq!(tp.total(), 3);
    }

    #[test]
    fn test_profile_data_type_at_missing() {
        let pd = ProfileData::new(1);
        assert!(pd.type_at(99).is_none());
    }

    #[test]
    fn test_profile_data_record_call() {
        let mut pd = ProfileData::new(1);
        pd.record_call(30, 100);
        pd.record_call(30, 100);
        pd.record_call(30, 200);
        let cp = pd.call_at(30).unwrap();
        assert_eq!(cp.total(), 3);
        assert_eq!(cp.target_count(), 2);
    }

    #[test]
    fn test_profile_data_call_at_missing() {
        let pd = ProfileData::new(1);
        assert!(pd.call_at(99).is_none());
    }

    #[test]
    fn test_profile_data_record_loop() {
        let mut pd = ProfileData::new(1);
        pd.record_loop_iteration(40);
        pd.record_loop_iteration(40);
        pd.record_loop_iteration(40);
        assert_eq!(pd.loop_count(40), 3);
        assert_eq!(pd.loop_count(99), 0);
    }

    #[test]
    fn test_profile_data_merge() {
        let mut pd1 = ProfileData::new(1);
        pd1.record_execution();
        pd1.record_branch(10, true);
        pd1.record_type(20, 1);
        pd1.record_call(30, 100);

        let mut pd2 = ProfileData::new(1);
        pd2.record_execution();
        pd2.record_execution();
        pd2.record_branch(10, false);
        pd2.record_type(20, 2);
        pd2.record_call(30, 200);

        pd1.merge(&pd2);
        assert_eq!(pd1.execution_count(), 3);
        let bp = pd1.branch_at(10).unwrap();
        assert_eq!(bp.taken, 1);
        assert_eq!(bp.not_taken, 1);
    }

    #[test]
    fn test_profile_data_scale() {
        let mut pd = ProfileData::new(1);
        for _ in 0..100 {
            pd.record_execution();
        }
        pd.record_branch(10, true);
        pd.record_branch(10, true);
        pd.record_branch(10, false);
        pd.record_branch(10, false);
        pd.scale(0.5);
        assert_eq!(pd.execution_count(), 50);
        let bp = pd.branch_at(10).unwrap();
        assert_eq!(bp.taken, 1);
        assert_eq!(bp.not_taken, 1);
    }

    // =========================================================================
    // Serialization Round-Trip Tests
    // =========================================================================

    #[test]
    fn test_serialize_empty() {
        let pd = ProfileData::new(42);
        let bytes = pd.serialize();
        let pd2 = ProfileData::deserialize(&bytes).unwrap();
        assert_eq!(pd2.code_id(), 42);
        assert_eq!(pd2.execution_count(), 0);
        assert_eq!(pd2.branch_count(), 0);
    }

    #[test]
    fn test_serialize_roundtrip_branches() {
        let mut pd = ProfileData::new(7);
        pd.record_branch(10, true);
        pd.record_branch(10, true);
        pd.record_branch(10, false);
        pd.record_branch(20, false);
        pd.record_branch(20, false);

        let bytes = pd.serialize();
        let pd2 = ProfileData::deserialize(&bytes).unwrap();

        assert_eq!(pd2.code_id(), 7);
        let bp10 = pd2.branch_at(10).unwrap();
        assert_eq!(bp10.taken, 2);
        assert_eq!(bp10.not_taken, 1);
        let bp20 = pd2.branch_at(20).unwrap();
        assert_eq!(bp20.taken, 0);
        assert_eq!(bp20.not_taken, 2);
    }

    #[test]
    fn test_serialize_roundtrip_types() {
        let mut pd = ProfileData::new(3);
        pd.record_type(10, 1);
        pd.record_type(10, 1);
        pd.record_type(10, 2);

        let bytes = pd.serialize();
        let pd2 = ProfileData::deserialize(&bytes).unwrap();

        let tp = pd2.type_at(10).unwrap();
        assert_eq!(tp.total(), 3);
        assert_eq!(tp.entries().len(), 2);
    }

    #[test]
    fn test_serialize_roundtrip_calls() {
        let mut pd = ProfileData::new(5);
        pd.record_call(30, 100);
        pd.record_call(30, 100);
        pd.record_call(30, 200);

        let bytes = pd.serialize();
        let pd2 = ProfileData::deserialize(&bytes).unwrap();

        let cp = pd2.call_at(30).unwrap();
        assert_eq!(cp.total(), 3);
        assert_eq!(cp.target_count(), 2);
    }

    #[test]
    fn test_serialize_roundtrip_execution_counts() {
        let mut pd = ProfileData::new(9);
        for _ in 0..1000 {
            pd.record_execution();
        }
        pd.record_loop_iteration(50);
        pd.record_loop_iteration(50);
        pd.record_loop_iteration(60);

        let bytes = pd.serialize();
        let pd2 = ProfileData::deserialize(&bytes).unwrap();

        assert_eq!(pd2.execution_count(), 1000);
        assert_eq!(pd2.loop_count(50), 2);
        assert_eq!(pd2.loop_count(60), 1);
    }

    #[test]
    fn test_serialize_roundtrip_complex() {
        let mut pd = ProfileData::new(99);
        for _ in 0..500 {
            pd.record_execution();
        }
        for i in 0..10u32 {
            pd.record_branch(i * 10, i % 2 == 0);
        }
        pd.record_type(100, 1);
        pd.record_type(100, 2);
        pd.record_type(200, 5);
        pd.record_call(300, 10);
        pd.record_call(300, 20);
        pd.record_loop_iteration(400);

        let bytes = pd.serialize();
        let pd2 = ProfileData::deserialize(&bytes).unwrap();

        assert_eq!(pd2.code_id(), 99);
        assert_eq!(pd2.execution_count(), 500);
        assert_eq!(pd2.branch_count(), 10);
        assert_eq!(pd2.type_count(), 2);
        assert_eq!(pd2.call_count(), 1);
        assert_eq!(pd2.loop_count(400), 1);
    }

    // =========================================================================
    // Deserialization Error Tests
    // =========================================================================

    #[test]
    fn test_deserialize_too_short() {
        let result = ProfileData::deserialize(&[0, 1, 2]);
        assert_eq!(result.unwrap_err(), ProfileError::TooShort);
    }

    #[test]
    fn test_deserialize_bad_magic() {
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(b"XXXX");
        let result = ProfileData::deserialize(&data);
        assert_eq!(result.unwrap_err(), ProfileError::BadMagic);
    }

    #[test]
    fn test_deserialize_bad_version() {
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(b"PPRF");
        data[4..6].copy_from_slice(&99u16.to_le_bytes());
        let result = ProfileData::deserialize(&data);
        assert_eq!(result.unwrap_err(), ProfileError::UnsupportedVersion(99));
    }

    #[test]
    fn test_deserialize_checksum_mismatch() {
        let pd = ProfileData::new(1);
        let mut bytes = pd.serialize();
        // Corrupt a byte after the header
        if bytes.len() > 18 {
            bytes[18] ^= 0xFF;
        }
        let result = ProfileData::deserialize(&bytes);
        assert_eq!(result.unwrap_err(), ProfileError::ChecksumMismatch);
    }

    #[test]
    fn test_profile_error_display() {
        assert_eq!(
            format!("{}", ProfileError::TooShort),
            "profile data too short"
        );
        assert_eq!(
            format!("{}", ProfileError::BadMagic),
            "invalid profile magic bytes"
        );
        assert_eq!(
            format!("{}", ProfileError::UnsupportedVersion(2)),
            "unsupported profile version: 2"
        );
        assert_eq!(
            format!("{}", ProfileError::ChecksumMismatch),
            "profile checksum mismatch"
        );
        assert_eq!(
            format!("{}", ProfileError::UnknownSection(99)),
            "unknown profile section: 99"
        );
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_branch_profile_extreme_bias() {
        let bp = BranchProfile::with_counts(1_000_000, 1);
        assert!(bp.is_biased());
        assert!((bp.taken_probability() - 0.999999).abs() < 0.001);
    }

    #[test]
    fn test_type_profile_single_observation_not_monomorphic() {
        let mut tp = TypeProfile::new();
        tp.record(1);
        assert!(!tp.is_monomorphic()); // Needs >=10 observations
    }

    #[test]
    fn test_call_profile_sorted_after_frequency_change() {
        let mut cp = CallProfile::new();
        for _ in 0..10 {
            cp.record(1);
        }
        for _ in 0..5 {
            cp.record(2);
        }
        // Target 1 should be first
        assert_eq!(cp.targets()[0].target_id, 1);

        // Now make target 2 more frequent
        for _ in 0..20 {
            cp.record(2);
        }
        // Target 2 should now be first
        assert_eq!(cp.targets()[0].target_id, 2);
        assert_eq!(cp.targets()[0].count, 25);
    }

    #[test]
    fn test_profile_data_multiple_offsets() {
        let mut pd = ProfileData::new(1);
        for offset in 0..100u32 {
            pd.record_branch(offset, offset % 2 == 0);
        }
        assert_eq!(pd.branch_count(), 100);
    }

    #[test]
    fn test_serialize_large_profile() {
        let mut pd = ProfileData::new(1);
        for i in 0..50u32 {
            for _ in 0..100 {
                pd.record_branch(i, true);
            }
            pd.record_type(i, (i % 8) as u8);
            pd.record_call(i, i * 10);
        }
        for _ in 0..10000 {
            pd.record_execution();
        }

        let bytes = pd.serialize();
        let pd2 = ProfileData::deserialize(&bytes).unwrap();

        assert_eq!(pd2.execution_count(), 10000);
        assert_eq!(pd2.branch_count(), 50);
        assert_eq!(pd2.type_count(), 50);
        assert_eq!(pd2.call_count(), 50);
    }

    #[test]
    fn test_serialize_deterministic() {
        let mut pd = ProfileData::new(1);
        pd.record_branch(10, true);
        pd.record_type(20, 1);
        pd.record_call(30, 100);
        pd.record_execution();

        let bytes1 = pd.serialize();
        let bytes2 = pd.serialize();
        assert_eq!(bytes1, bytes2);
    }
}
