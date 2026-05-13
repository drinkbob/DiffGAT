import os
import sys
import multiprocessing
from gooey import Gooey, GooeyParser

@Gooey(
    program_name="DiffGAT 分子生成与分析工具",
    program_description="集成了模型训练、消融实验、基准测试与多核生成的图形化流水线",
    default_size=(850, 650),
    clear_before_run=True # 每次运行前清空控制台
)
def main():
    parser = GooeyParser(description="请选择左侧栏的功能菜单并填写相关参数")
    subs = parser.add_subparsers(dest="mode", required=True)

    # 1. train 子命令
    train_parser = subs.add_parser("train", help="模型训练 (支持核心骨架约束)")
    train_parser.add_argument("epochs", type=int, default=50, help="训练轮数")
    train_parser.add_argument("csv_path", widget="FileChooser", help="训练集 CSV 文件 (可选，不选则使用默认内置数据)", nargs='?', default="")

    # 2. ablation 子命令
    ablation_parser = subs.add_parser("ablation", help="消融实验评估")
    ablation_parser.add_argument("num_molecules", type=int, default=100, help="生成分子数")
    ablation_parser.add_argument("complexity", choices=["simple", "medium", "complex"], default="medium", help="复杂度")
    ablation_parser.add_argument("--train_csv", widget="FileChooser", help="训练集 CSV (用于计算 Novelty, 必须提供)")

    # 3. benchmark 子命令
    bench_parser = subs.add_parser("benchmark", help="基准测试与对比")
    bench_parser.add_argument("--num", type=int, default=500, help="样本数")
    bench_parser.add_argument("--ours_csv", widget="FileChooser", help="DiffGAT 输出的 CSV 结果文件")
    bench_parser.add_argument("--baseline_csv", nargs=2, action="append", help="外部基线 CSV (填法：模型名称 对应CSV文件)")

    # 4. multicore 子命令
    mc_parser = subs.add_parser("multicore", help="多核心分子生成")
    mc_parser.add_argument("--per_core", type=int, default=30, help="每个核心生成的分子数")
    mc_parser.add_argument("--complexity", choices=["simple", "medium", "complex"], default="medium", help="复杂度")
    mc_parser.add_argument("--seed", type=int, default=42, help="随机种子 (保证可重复性)")

    args = parser.parse_args()

    # ==========================
    # 模式分发逻辑 (带安全防护)
    # ==========================
    
    if args.mode == "train":
        from improved_hybrid_molecular_generator import train_improved_hybrid_generator
        
        # 安全处理路径为空的情况
        safe_csv_path = args.csv_path if args.csv_path else None
        print(f"正在启动训练任务... 轮数: {args.epochs}")
        train_improved_hybrid_generator(epochs=args.epochs, custom_csv_path=safe_csv_path)

    elif args.mode == "ablation":
        from ablation_evaluation import main as ablation_main
        
        # 修复点 2：防止 NoneType 污染 sys.argv
        new_argv = [sys.argv[0], str(args.num_molecules), args.complexity]
        if args.train_csv:
            new_argv.extend(["--train_csv", args.train_csv])
        
        sys.argv = new_argv
        ablation_main()

    elif args.mode == "benchmark":
        from baseline_benchmark import main as bench_main
        
        new_argv = [sys.argv[0]]
        if args.num:
            new_argv.extend(["--num", str(args.num)])
            
        if args.ours_csv:
            new_argv.extend(["--ours_csv", args.ours_csv])
            
        if args.baseline_csv:
            for pair in args.baseline_csv:
                # pair 结构为 [ModelName, FilePath]
                if pair[0] and pair[1]:
                    new_argv.extend(["--baseline_csv", pair[0], pair[1]])
        else:
            # 如果用户没提供外部 baselines 进行严格对比，强制开启内部 fallback，防止程序直接崩溃退出
            print("注意：未提供外部基线 CSV 文件，已自动切换为内置 fallback 生成模式。")
            new_argv.append("--legacy_proxy_mode")
            new_argv.append("--allow_fallback_baselines")

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
    multiprocessing.freeze_support()
    main()
