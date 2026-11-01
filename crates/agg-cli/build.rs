fn main() {
    println!("cargo:rerun-if-env-changed=AGG_GIT_REV");
}
