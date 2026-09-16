{
  description = "Dev environment for sales-forecast (demand forecasting, Streamlit app)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = { self, nixpkgs, flake-parts, ... }@inputs:
    flake-parts.lib.mkFlake { inherit self; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];

      perSystem = { system, ... }: {
        devShells.default =
          let
            pkgs = nixpkgs.legacyPackages.${system};
          in
          pkgs.mkShell {
            packages = with pkgs; [
              uv          # zarządzanie środowiskiem i zależnościami Pythona (z uv.lock)
              git
            ];

            shellHook = ''
              echo "[sales-forecast] nix devShell: uv $(uv --version | cut -d' ' -f2)"
              echo "[sales-forecast] uruchomienie: uv sync && uv run streamlit run src/app.py"
            '';
          };
      };
    };
}
