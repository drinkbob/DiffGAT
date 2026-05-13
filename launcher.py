import os
import sys
from gooey import Gooey, GooeyParser

@Gooey(
    program_name="DiffGAT 分子生成工具",
    program_description="训练、消融、基准测试、多核生成",
    default_size=(700, 600)
)
def main():
    parser = GooeyParser(description="选择一个功能运行")
    subs = parser.add_subparsers(dest="mode", required=True)

    # train 子命令
    train_parser = subs.add_parser("train", help="训练模型")
    train_parser.add_argument("epochs", type=int, help="训练轮数")
    train_parser.add_argument("csv_path", widget="FileChooser", help="训练集 CSV 文件")

    # ablation 子命令
    ablation_parser = subs.add_parser("ablation", help="消融实验")
    ablation_parser.add_argument("num_molecules", type=int, help="生成分子数")
    ablation_parser.add_argument("complexity", choices=["simple", "medium", "complex"], help="复杂度")
    ablation_parser.add_argument("train_csv", widget="FileChooser", help="训练集 CSV (用于计算 Novelty)")

    # benchmark 子命令
    bench_parser = subs.add_parser("benchmark", help="基准测试")
    bench_parser.add_argument("--num", type=int, default=500, help="样本数")
    bench_parser.add_argument("--ours_csv", widget="FileChooser", help="DiffGAT 输出 CSV")
    bench_parser.add_argument("--baseline_csv", nargs=2, action="append", help="外部基线 CSV (模型名 文件)")

    # multicore 子命令
    mc_parser = subs.add_parser("multicore", help="多核生成")
    mc_parser.add_argument("--per_core", type=int, default=30, help="每个核心生成的分子数")
    mc_parser.add_argument("--complexity", choices=["simple", "medium", "complex"], default="medium")
    mc_parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    if args.mode == "train":
        from improved_hybrid_molecular_generator import main as train_main
        sys.argv = [sys.argv[0], str(args.epochs), args.csv_path]
        train_main()

    elif args.mode == "ablation":
        from ablation_evaluation import main as ablation_main
        sys.argv = [sys.argv[0], str(args.num_molecules), args.complexity, "--train_csv", args.train_csv]
        ablation_main()

    elif args.mode == "benchmark":
        from baseline_benchmark import main as bench_main
        # 构建命令行参数
        new_argv = [sys.argv[0]]
        if args.num:
            new_argv += ["--num", str(args.num)]
        if args.ours_csv:
            new_argv += ["--ours_csv", args.ours_csv]
        if args.baseline_csv:
            for pair in args.baseline_csv:
                new_argv += ["--baseline_csv"] + list(pair)
        sys.argv = new_argv
        bench_main()

    elif args.mode == "multicore":
        from multi_core_molecular_generator import main as mc_main
        new_argv = [sys.argv[0],
                    "--per_core", str(args.per_core),
                    "--complexity", args.complexity,
                    "--seed", str(args.seed)]
        sys.argv = new_argv
        mc_main()

if __name__ == "__main__":
    main()
