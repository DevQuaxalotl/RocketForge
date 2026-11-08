fn main() {
    let root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let output = std::env::var("OUT_DIR").unwrap();
    let config = cbindgen::Config::from_file(format!("{root}/cbindgen.toml")).unwrap();
    cbindgen::Builder::new()
        .with_crate(&root)
        .with_config(config)
        .generate()
        .expect("generate C header")
        .write_to_file(format!("{output}/rocketsim_ffi.h"));
    let generated = std::fs::read_to_string(format!("{output}/rocketsim_ffi.h")).unwrap();
    let checked_in = std::fs::read_to_string(format!("{root}/include/rocketsim_ffi.h")).unwrap();
    assert_eq!(generated.replace("\r\n", "\n"), checked_in.replace("\r\n", "\n"),
        "Generated C header changed. Copy OUT_DIR/rocketsim_ffi.h into bindings/include before rebuilding C++.");
    println!("cargo:rerun-if-changed=include/rocketsim_ffi.h");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
