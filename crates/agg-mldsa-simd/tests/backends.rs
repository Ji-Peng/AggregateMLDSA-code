mod common;

#[test]
fn test_backends_stub() {
    let engines = common::engines();
    assert!(!engines.is_empty());
}
