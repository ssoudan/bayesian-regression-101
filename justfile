targets := "bayesian-regression-101-core" 
extra_features := ""

default: fmt check clippy test build

test:
    @echo "Running tests..."
    @cargo test --all --all-features

check:
    @echo "Checking..."
    @cargo check --all --all-features --tests --examples --benches

clippy:
    @echo "Running clippy..."
    @cargo clippy --all --all-features --tests --examples --benches

fmt:
    @echo "Formatting..."
    @cargo +nightly fmt --all --check

machete:
    @echo "Running machete..."
    @cargo +nightly machete

deny:
    @echo "Running deny..."
    @cargo +nightly deny check

build:
    @echo "Building..."
    @cargo build --all --all-features

benches:
    @echo "Running benches..."
    @cargo bench --all --features=unstable

doc $RUSTDOCFLAGS="-D warnings":
    @echo "Building docs..."    
    @cargo doc --no-deps --document-private-items --all-features --workspace --examples

update: check && check clippy fmt machete build
    @echo "Updating..."
    @cargo update
    @cargo upgrade

docker:
    @echo "Building docker image..."
    
    for t in {{targets}}; do \    
    docker build --target $t -t $t --build-arg EXTRA_FEATURES="{{extra_features}}" . ; \
    done
    
