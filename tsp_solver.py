import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import random
import math
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


class TSPSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Solver - OR-Tools")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f0f0f0")

        self.solving = False
        self.solution_data = None
        self.nodes = []

        self._build_ui()

    def _build_ui(self):
        # --- 左パネル（コントロール） ---
        left = tk.Frame(self.root, bg="#f0f0f0", width=260)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=12)
        left.pack_propagate(False)

        tk.Label(left, text="TSP Solver", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=(0, 16))

        # 拠点数スライダー
        tk.Label(left, text="拠点数", font=("Arial", 11), bg="#f0f0f0").pack(anchor=tk.W)
        self.node_var = tk.IntVar(value=20)
        self.node_slider = tk.Scale(
            left, from_=5, to=6000, orient=tk.HORIZONTAL,
            variable=self.node_var, length=220,
            command=self._on_slider_change,
            bg="#f0f0f0", highlightthickness=0
        )
        self.node_slider.pack(fill=tk.X)
        self.node_label = tk.Label(left, text="20 拠点", font=("Arial", 11, "bold"), bg="#f0f0f0", fg="#333")
        self.node_label.pack()

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # タイムリミット
        tk.Label(left, text="計算時間上限 (秒)", font=("Arial", 11), bg="#f0f0f0").pack(anchor=tk.W)
        self.timelimit_var = tk.IntVar(value=30)
        self.timelimit_slider = tk.Scale(
            left, from_=5, to=120, orient=tk.HORIZONTAL,
            variable=self.timelimit_var, length=220,
            bg="#f0f0f0", highlightthickness=0
        )
        self.timelimit_slider.pack(fill=tk.X)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 求解モード切替
        tk.Label(left, text="求解モード", font=("Arial", 11), bg="#f0f0f0").pack(anchor=tk.W)
        self.mode_var = tk.StringVar(value="greedy")
        mode_frame = tk.Frame(left, bg="#f0f0f0")
        mode_frame.pack(fill=tk.X, pady=2)
        self.radio_greedy = tk.Radiobutton(
            mode_frame, text="貪欲法モード（最近傍法）",
            variable=self.mode_var, value="greedy",
            command=self._on_mode_change,
            bg="#f0f0f0", font=("Arial", 10), activebackground="#f0f0f0"
        )
        self.radio_greedy.pack(anchor=tk.W)
        self.radio_gls = tk.Radiobutton(
            mode_frame, text="GLSモード（近似・高速）",
            variable=self.mode_var, value="gls",
            command=self._on_mode_change,
            bg="#f0f0f0", font=("Arial", 10), activebackground="#f0f0f0"
        )
        self.radio_gls.pack(anchor=tk.W)
        self.mode_desc = tk.Label(left, text="貪欲法：拠点数ぶんのステップで進捗表示", font=("Arial", 9), bg="#f0f0f0", fg="#888")
        self.mode_desc.pack(anchor=tk.W)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # ボタン
        self.generate_btn = tk.Button(
            left, text="拠点を生成", font=("Arial", 11),
            command=self.generate_nodes, bg="#4CAF50", fg="white",
            relief=tk.FLAT, padx=10, pady=6, cursor="hand2"
        )
        self.generate_btn.pack(fill=tk.X, pady=4)

        self.solve_btn = tk.Button(
            left, text="求解スタート", font=("Arial", 11),
            command=self.start_solving, bg="#2196F3", fg="white",
            relief=tk.FLAT, padx=10, pady=6, cursor="hand2"
        )
        self.solve_btn.pack(fill=tk.X, pady=4)

        self.stop_btn = tk.Button(
            left, text="停止", font=("Arial", 11),
            command=self.stop_solving, bg="#f44336", fg="white",
            relief=tk.FLAT, padx=10, pady=6, cursor="hand2",
            state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=4)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 結果表示
        tk.Label(left, text="結果", font=("Arial", 11, "bold"), bg="#f0f0f0").pack(anchor=tk.W)
        self.result_frame = tk.Frame(left, bg="#ffffff", relief=tk.SUNKEN, bd=1)
        self.result_frame.pack(fill=tk.X, pady=4)

        labels = [
            ("状態", "status_val", "待機中"),
            ("拠点数", "nodes_val", "-"),
            ("総距離", "distance_val", "-"),
            ("計算時間", "time_val", "-"),
        ]
        for text, attr, default in labels:
            row = tk.Frame(self.result_frame, bg="#ffffff")
            row.pack(fill=tk.X, padx=6, pady=2)
            tk.Label(row, text=text + ":", width=8, anchor=tk.W, bg="#ffffff", font=("Arial", 10)).pack(side=tk.LEFT)
            val_label = tk.Label(row, text=default, anchor=tk.W, bg="#ffffff", font=("Arial", 10, "bold"), fg="#333")
            val_label.pack(side=tk.LEFT)
            setattr(self, attr, val_label)

        # プログレスバー
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        tk.Label(left, text="進行状況", font=("Arial", 11), bg="#f0f0f0").pack(anchor=tk.W)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            left, variable=self.progress_var, maximum=100,
            mode="determinate", length=220
        )
        self.progress_bar.pack(fill=tk.X, pady=4)
        self.progress_label = tk.Label(left, text="待機中", font=("Arial", 10), bg="#f0f0f0", fg="#666")
        self.progress_label.pack(anchor=tk.W)

        # --- 右パネル（キャンバス） ---
        right = tk.Frame(self.root, bg="#ffffff", relief=tk.SUNKEN, bd=1)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 12), pady=12)

        self.canvas = tk.Canvas(right, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 初期表示（ウィンドウ描画後にサイズが確定してから実行）
        self.root.after(100, self.generate_nodes)

    def _on_slider_change(self, val):
        self.node_label.config(text=f"{val} 拠点")

    def _on_mode_change(self):
        if self.mode_var.get() == "greedy":
            self.mode_desc.config(text="貪欲法：拠点数ぶんのステップで進捗表示")
            self.timelimit_slider.config(state=tk.DISABLED)
        else:
            self.mode_desc.config(text="GLS：貪欲解を初期解としてOR-Toolsで改善")
            self.timelimit_slider.config(state=tk.NORMAL)

    def generate_nodes(self):
        if self.solving:
            return
        n = self.node_var.get()
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width() or 700
        h = self.canvas.winfo_height() or 650
        pad = 40
        self.nodes = [(random.randint(pad, w - pad), random.randint(pad, h - pad)) for _ in range(n)]
        self.solution_data = None
        self._reset_results()
        self._draw(route=None)

    def _reset_results(self):
        self.status_val.config(text="待機中", fg="#333")
        self.nodes_val.config(text=str(len(self.nodes)))
        self.distance_val.config(text="-")
        self.time_val.config(text="-")
        self.progress_var.set(0)
        self.progress_label.config(text="待機中")

    def _draw(self, route=None):
        self.canvas.delete("all")
        if not self.nodes:
            return

        # ルート描画
        if route:
            for i in range(len(route) - 1):
                x1, y1 = self.nodes[route[i]]
                x2, y2 = self.nodes[route[i + 1]]
                self.canvas.create_line(x1, y1, x2, y2, fill="#2196F3", width=2)

        # ノード描画
        r = max(2, 5 - len(self.nodes) // 200)
        for i, (x, y) in enumerate(self.nodes):
            color = "#E53935" if i == 0 else "#4CAF50"
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="white", width=1)
            if len(self.nodes) <= 50:
                self.canvas.create_text(x + 8, y - 8, text=str(i), font=("Arial", 7), fill="#555")

    def start_solving(self):
        if self.solving or not self.nodes:
            if not self.nodes:
                messagebox.showwarning("警告", "先に拠点を生成してください")
            return
        self.solving = True
        self.solve_btn.config(state=tk.DISABLED)
        self.generate_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_val.config(text="計算中...", fg="#FF9800")
        self._set_progress(0, "開始中...")
        self._stop_flag = False
        self._progress_stage = "matrix"
        self._poll_start = time.perf_counter()
        self._poll_use_gls = self.mode_var.get() == "gls"
        self._poll_time_limit = self.timelimit_var.get()
        n = len(self.nodes)
        # 距離行列構築の推定時間（経験則：n²/1e7 秒）
        self._matrix_est = max(0.5, (n * n) / 1e7)
        self._solve_start = time.perf_counter()
        self._set_progress(0, "開始中...")
        self._solve_thread = threading.Thread(target=self._solve_worker, daemon=True)
        self._solve_thread.start()
        self.root.after(200, self._poll_progress)

    def stop_solving(self):
        self._stop_flag = True

    def _solve_worker(self):
        n = len(self.nodes)
        time_limit = self.timelimit_var.get()
        use_gls = self.mode_var.get() == "gls"
        start = time.perf_counter()

        # --- 距離行列構築（numpy のまま保持）---
        self._progress_stage = "matrix"
        coords = np.array(self.nodes, dtype=np.float32)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_np = np.sqrt((diff ** 2).sum(axis=2)).astype(np.int32)

        if self._stop_flag:
            self.root.after(0, self._on_stopped)
            return

        # --- 貪欲法（最近傍法）で初期ルートを構築 ---
        # 進捗：1ステップ = 1/n ずつ正確に進む
        self._progress_stage = "greedy"
        self._greedy_n = n
        self._greedy_step = 0

        visited = np.zeros(n, dtype=bool)
        route = [0]
        visited[0] = True
        current = 0

        for step in range(n - 1):
            if self._stop_flag:
                self.root.after(0, self._on_stopped)
                return
            row = dist_np[current].copy()
            row[visited] = np.iinfo(np.int32).max
            nxt = int(np.argmin(row))
            route.append(nxt)
            visited[nxt] = True
            current = nxt
            self._greedy_step = step + 1

        route.append(0)  # スタート地点に戻る

        # 貪欲法の総距離
        greedy_dist = int(sum(dist_np[route[i], route[i + 1]] for i in range(n)))

        elapsed_total = time.perf_counter() - start

        if not use_gls:
            # 貪欲法モード：そのまま返す
            self.root.after(0, lambda: self._set_progress(100, "完了"))
            self.root.after(0, lambda: self._on_solved(route, greedy_dist, elapsed_total))
            return

        # --- GLSモード：貪欲解を初期解として OR-Tools で改善 ---
        self._progress_stage = "gls"
        self._solve_start = time.perf_counter()
        self._poll_time_limit = time_limit

        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        # numpy配列を直接参照するコールバック（list より高速）
        def distance_callback(from_idx, to_idx):
            return int(dist_np[manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)])

        transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_params.time_limit.seconds = time_limit

        # 貪欲解を初期解として登録
        initial_routes = [route[:-1]]
        assignment = routing.ReadAssignmentFromRoutes(initial_routes, True)
        solution = routing.SolveFromAssignmentWithParameters(assignment, search_params)

        elapsed_total = time.perf_counter() - start

        if self._stop_flag:
            self.root.after(0, self._on_stopped)
            return

        self.root.after(0, lambda: self._set_progress(100, "完了"))

        if solution:
            gls_route = []
            idx = routing.Start(0)
            while not routing.IsEnd(idx):
                gls_route.append(manager.IndexToNode(idx))
                idx = solution.Value(routing.NextVar(idx))
            gls_route.append(manager.IndexToNode(idx))
            total_dist = solution.ObjectiveValue()
            self.root.after(0, lambda: self._on_solved(gls_route, total_dist, elapsed_total))
        else:
            # GLS失敗時は貪欲解で返す
            self.root.after(0, lambda: self._on_solved(route, greedy_dist, elapsed_total))

    def _set_progress(self, pct, msg=""):
        """メインスレッドから直接呼ぶ用（root.after経由）"""
        self.progress_var.set(pct)
        self.progress_label.config(text=f"{msg}")

    def _poll_progress(self):
        """ワーカースレッドの進捗をメインスレッドで定期的に反映"""
        if self.solving:
            elapsed = time.perf_counter() - self._poll_start
            stage = getattr(self, "_progress_stage", "matrix")

            if stage == "matrix":
                est = max(0.5, self._matrix_est)
                pct = min(10, (elapsed / est) * 10)
                self._set_progress(pct, f"距離行列を構築中... ({elapsed:.1f}s)")

            elif stage == "greedy":
                n = getattr(self, "_greedy_n", 1)
                step = getattr(self, "_greedy_step", 0)
                pct = 10 + (step / n) * 80  # 10%→90%
                self._set_progress(pct, f"貪欲法探索中... {step} / {n} 拠点")

            elif stage == "gls":
                time_limit = self._poll_time_limit
                solve_elapsed = time.perf_counter() - self._solve_start
                pct = min(90 + (solve_elapsed / time_limit) * 9, 98)
                self._set_progress(pct, f"GLS改善中... {solve_elapsed:.1f}s / {time_limit}s")

            self.root.after(200, self._poll_progress)

    def _on_solved(self, route, total_dist, elapsed):
        self.solving = False
        self._finalize_ui()
        self.status_val.config(text="求解完了", fg="#4CAF50")
        self.distance_val.config(text=f"{total_dist:,}")
        self.time_val.config(text=f"{elapsed:.3f} 秒")
        self._draw(route=route)

    def _on_no_solution(self, elapsed):
        self.solving = False
        self._finalize_ui()
        self.status_val.config(text="解なし", fg="#f44336")
        self.time_val.config(text=f"{elapsed:.3f} 秒")

    def _on_stopped(self):
        self.solving = False
        self._finalize_ui()
        self.status_val.config(text="停止", fg="#FF9800")
        self._set_progress(0, "停止しました")

    def _finalize_ui(self):
        self.solve_btn.config(state=tk.NORMAL)
        self.generate_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = TSPSolverApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
