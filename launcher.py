import os
import sys

def main():
    if len(sys.argv) < 2:
        print("用法：")
        print("  DiffGAT.exe train <epochs> <csv_path>")
        print("  DiffGAT.exe ablation <num_molecules> <complexity> <train_csv>")
        print("  DiffGAT.exe benchmark --num 500 ...")
        print("  DiffGAT.exe multicore --input_csv ...")
        sys.exit(1)

    mode = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if mode == "train":
        from improved_hybrid_molecular_generator import main as train_main
        train_main()
    elif mode == "ablation":
        from ablation_evaluation import main as ablation_main
        ablation_main()
    elif mode == "benchmark":
        from baseline_benchmark import main as bench_main
        bench_main()
    elif mode == "multicore":
        from multi_core_molecular_generator import main as mc_main
        mc_main()
    else:
        print(f"未知模式: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
