{
  description = "Software defined LaserDisc decoder";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pname = "vhs-decode";
        
        python = pkgs.python312;
        pythonPackages = python.pkgs;
        
        # PEP-440 compatible version string (for package metadata)
        version = "3.10.0";
        
        # Use flake's built-in git properties
        # dirtyShortRev already includes "-dirty" suffix, so we need to handle it
        gitCommit = if self ? dirtyShortRev then self.dirtyShortRev else self.shortRev;
        
        # Build PEP-440 compliant version string with git info
        # Format: base_version+git.commit[.dirty]
        # dirtyShortRev format is "abc1234-dirty", so replace "-" with "."
        fullVersion = "${version}+git.${builtins.replaceStrings ["-"] ["."] gitCommit}";
        
        docsEnv = python.withPackages (ps: with ps; [
          mkdocs
          mkdocs-material
          mkdocs-awesome-nav
        ]);

        cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
          inherit pname version;
          src = ./.;
          hash = "sha256-yryE7R0A95Uok6Pv6/UBsIG8p9pvaP3Nv8AGQugrEOc=";
        };

        vhs-decode = pythonPackages.buildPythonPackage {
          inherit pname version cargoDeps;
          
          src = ./.;
          
          pyproject = true;
          
          nativeBuildInputs = with pythonPackages; [
            pkgs.cargo
            pkgs.rustPlatform.cargoSetupHook
            setuptools
            setuptools-rust
            setuptools-scm
            wheel
            cython
            pkgs.rustc
          ];
          
          propagatedBuildInputs = with pythonPackages; [
            av
            cython
            matplotlib
            noisereduce
            numba
            numpy
            scipy
            setproctitle
            sounddevice
            soundfile
            soxr
          ];
          
          # static-ffmpeg is not in nixpkgs; ffmpeg is provided via pkgs.ffmpeg
          postPatch = ''
            substituteInPlace pyproject.toml \
              --replace-fail '    "static-ffmpeg",' ""
          '';

          # Write PEP-440 compliant version file with git info
          preBuild = ''
            echo "${fullVersion}" > lddecode/version
          '';
          
          # Skip tests for minimal build
          doCheck = false;
          
          meta = with pkgs.lib; {
            description = "Software defined LaserDisc and videotape decoder";
            homepage = "https://github.com/oyvindln/vhs-decode";
            license = licenses.gpl3Plus;
          };
        };
      in
      {
        packages = {
          default = vhs-decode;
          vhs-decode = vhs-decode;
          docs = pkgs.stdenv.mkDerivation {
            pname = "ld-decode-docs";
            inherit version;
            src = ./.;
            nativeBuildInputs = [ docsEnv ];
            buildPhase = ''mkdocs build'';
            installPhase = ''cp -r site $out'';
          };
        };
        
        apps = {
          default = {
            type = "app";
            program = "${vhs-decode}/bin/vhs-decode";
          };
          vhs-decode = {
            type = "app";
            program = "${vhs-decode}/bin/vhs-decode";
          };
          cvbs-decode = {
            type = "app";
            program = "${vhs-decode}/bin/cvbs-decode";
          };
          ld-decode = {
            type = "app";
            program = "${vhs-decode}/bin/ld-decode";
          };
          ld-ldf-reader-py = {
            type = "app";
            program = "${vhs-decode}/bin/ld-ldf-reader-py";
          };
        };
        
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.cmake
            pkgs.ffmpeg
            vhs-decode
            pythonPackages.jupyter
            pythonPackages.pandas
            pythonPackages.pytest
            pythonPackages.pytest-cov
            docsEnv
          ];
          
          shellHook = ''
            echo "vhs-decode development environment"
          '';
        };
      }
    );
}
