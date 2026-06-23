"""
応急危険度判定 mTSP - 時間制約付き複数巡回セールスマン問題
Time-Constrained mTSP for Emergency Building Inspection

【概要】
地震等の災害後に複数の判定士が拠点（デポ）から出発し、
エリア内の全建物を手分けして応急危険度判定を行う巡回計画を作成する。

【アルゴリズム】
- 最近傍法 (Nearest Neighbor) + KDTree による O(n log n) 近傍探索
- min-heap によるラウンドロビン割当（makespan 最小化）
- ProcessPoolExecutor による並列試行（複数デポ候補から最短を採用）
- 最小判定士数は二分探索で O(log n) 回の試行で確定

【制約条件】
1. 各建物は1回だけ判定（visited フラグ）
2. 判定士は拠点から出発・拠点に帰還
3. 移動時間（距離 / 速度）を稼働時間に加算
4. 判定時間（建物ごとに設定）を稼働時間に加算
5. 現在時刻 + 移動 + 判定 + デポ帰還 ≤ 最大稼働時間 を満たす場合のみ割当
6. 目的関数: 最大終了時間（makespan）の最小化

【依存ライブラリ】
    pip install ortools scipy matplotlib numpy
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
import random
import heapq
import numpy as np
from scipy.spatial import KDTree
import matplotlib
matplotlib.rcParams['font.family'] = 'Meiryo'
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from concurrent.futures import ProcessPoolExecutor, as_completed


# 判定士ごとのルート描画色（最大20人分）
COLORS = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4",
    "#8bc34a","#ff5722","#607d8b","#795548","#9c27b0",
    "#03a9f4","#cddc39","#ff9800","#673ab7","#009688",
]


# ── コアアルゴリズム（ProcessPoolExecutor で並列実行するためトップレベルに定義）──

def _mtsp_worker(args):
    """
    時間制約付き mTSP 貪欲法ワーカー。

    【割当戦略】
    min-heap で「現在時刻が最も小さい（最も暇な）判定士」が次の建物を選ぶ。
    これにより各判定士の終了時刻が平準化され、makespan が最小化される。

    【実行フロー】
    1. 全判定士を時刻 0・デポ位置でヒープに積む
    2. ヒープから最小時刻の判定士 s を取り出す
    3. KDTree で現在地に近い未訪問建物を候補として取得
    4. 制約チェック: t_now + 移動時間 + 判定時間 + デポ帰還時間 ≤ max_work_sec
    5. 条件を満たす最近傍を割当 → ヒープに再投入
    6. どの建物も割当不可なら担当者 s は終了（done フラグ）
    7. 全担当者終了 or 全建物割当完了で終了

    Args:
        args: (coords_arr, inspect_times, depot_idx, m, area_km, speed_kmh, max_work_h)

    Returns:
        (makespan, total_dist, routes, per_time, per_dist, unassigned)
        - makespan   : 最大終了時間 [時間]
        - total_dist : 総移動距離 [km]
        - routes     : 判定士ごとの訪問順インデックスリスト
        - per_time   : 判定士ごとの稼働時間 [時間]
        - per_dist   : 判定士ごとの移動距離 [km]
        - unassigned : 時間不足で割当できなかった建物数
    """
    (coords_arr, inspect_times, depot_idx,
     m, area_km, speed_kmh, max_work_h) = args

    n = len(coords_arr)
    max_work_sec = max_work_h * 3600.0        # 最大稼働時間を秒換算
    speed_ms = speed_kmh * 1000 / 3600.0      # 移動速度を m/s 換算
    area_m   = area_km * 1000.0               # エリアサイズを m 換算

    def travel_sec(i, j):
        """2点間の移動時間（秒）= ユークリッド距離 × エリアスケール / 速度"""
        d = float(np.linalg.norm(coords_arr[i] - coords_arr[j]))
        return d * area_m / speed_ms

    # KDTree を構築（近傍探索を O(log n) で実現）
    tree = KDTree(coords_arr)

    visited = np.zeros(n, dtype=bool)
    visited[depot_idx] = True   # デポは最初から訪問済み扱い

    # min-heap: (現在時刻[秒], 判定士ID)
    heap = [(0.0, s) for s in range(m)]
    heapq.heapify(heap)

    routes   = [[depot_idx] for _ in range(m)]  # 各判定士のルート（デポ始点）
    cur_pos  = [depot_idx] * m                  # 各判定士の現在位置
    cur_time = [0.0] * m                        # 各判定士の経過時間[秒]
    done     = [False] * m                      # これ以上割当不可フラグ

    unassigned = 0

    # 初回クエリの近傍数。未訪問が見つからなければ段階的に拡大
    k_base = min(30, n)

    remaining = n - 1   # デポを除いた未割当建物数

    while remaining > 0:
        if all(done):
            # 全判定士が稼働時間不足で終了 → 残りは未割当
            unassigned = remaining
            break

        cur_t, s = heapq.heappop(heap)
        if done[s]:
            continue

        pos   = cur_pos[s]
        t_now = cur_time[s]

        # 近傍を段階的に拡大しながら割当可能な建物を探す
        found = False
        for k_try in [k_base, k_base * 5, n]:
            k_try = min(k_try, n)
            _, idxs = tree.query(coords_arr[pos], k=k_try)
            idxs = np.atleast_1d(idxs)
            for nxt in idxs:
                if visited[nxt]:
                    continue
                t_travel  = travel_sec(pos, int(nxt))
                t_inspect = inspect_times[nxt]        # この建物の判定時間
                t_back    = travel_sec(int(nxt), depot_idx)  # デポへの帰還時間

                # 制約チェック: 建物訪問 + 帰還後も最大稼働時間内か
                if t_now + t_travel + t_inspect + t_back <= max_work_sec:
                    routes[s].append(int(nxt))
                    visited[nxt]  = True
                    cur_pos[s]    = int(nxt)
                    cur_time[s]   = t_now + t_travel + t_inspect
                    remaining    -= 1
                    heapq.heappush(heap, (cur_time[s], s))
                    found = True
                    break
            if found:
                break

        if not found:
            # どの未訪問建物も時間制約を満たせない → この判定士は終了
            done[s] = True

    # 全判定士をデポに帰還させ、稼働時間・移動距離を集計
    per_time = []
    per_dist = []
    for s in range(m):
        routes[s].append(depot_idx)
        t_back  = travel_sec(cur_pos[s], depot_idx)
        total_s = cur_time[s] + t_back
        per_time.append(total_s / 3600.0)   # 秒 → 時間

        r = routes[s]
        d = sum(
            float(np.linalg.norm(coords_arr[r[i]] - coords_arr[r[i+1]])) * area_km
            for i in range(len(r) - 1)
        )
        per_dist.append(d)

    makespan   = max(per_time)
    total_dist = sum(per_dist)
    return makespan, total_dist, routes, per_time, per_dist, unassigned


def _parallel_mtsp(coords, inspect_times, depot_indices,
                   m, area_km, speed_kmh, max_work_h, n_workers):
    """
    複数のデポ候補で _mtsp_worker を並列実行し、makespan 最小の結果を返す。

    デポ位置によってルートの質が変わるため、複数候補を並列試行することで
    解の質を向上させる（greedy の確率的改善）。
    """
    args_list = [
        (coords, inspect_times, d, m, area_km, speed_kmh, max_work_h)
        for d in depot_indices
    ]
    best = None
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_mtsp_worker, a): a for a in args_list}
        for f in as_completed(futs):
            res = f.result()
            # makespan 優先、同値なら総距離で比較
            if best is None or (res[0], res[1]) < (best[0], best[1]):
                best = res
    return best  # (makespan, total_dist, routes, per_time, per_dist, unassigned)


# ── GUI ───────────────────────────────────────────────────────────────────────

class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("応急危険度判定 mTSP — 時間制約付き巡回計画")
        self.root.geometry("1420x900")

        self.nodes        = []    # 建物座標リスト [(x, y), ...]  座標系: [0,1]x[0,1]
        self.inspect_times= []    # 建物ごとの判定時間 [秒]
        self.depot_idx    = 0     # デポの建物インデックス
        self.routes       = []    # 判定士ごとの訪問順インデックスリスト
        self.per_time     = []    # 判定士ごとの稼働時間 [時間]
        self.per_dist     = []    # 判定士ごとの移動距離 [km]
        self.unassigned   = 0     # 未割当建物数
        self.solving      = False
        self.n_cpu        = os.cpu_count() or 4
        self.depot_mode_var = tk.StringVar(value="center")

        self._build_ui()

    # ── UI 構築 ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # 左側コントロールパネル
        ctrl = tk.Frame(self.root, width=310, bg="#f0f0f0", padx=10, pady=8)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="応急危険度判定 mTSP",
                 font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=(0,3))
        tk.Label(ctrl, text=f"CPU: {self.n_cpu} コア",
                 bg="#f0f0f0", fg="#888", font=("Arial", 8)).pack(anchor="w")

        # 建物生成セクション
        self._sep(ctrl, "建物生成")
        self._row(ctrl, "建物数:",            "n_var",    "500", 9)
        self._row(ctrl, "エリア (km):",       "area_var", "10",  6)
        self._row(ctrl, "判定時間 最小(分):", "tmin_var", "15",  6)
        self._row(ctrl, "判定時間 最大(分):", "tmax_var", "45",  6)
        tk.Label(ctrl, text="建物ごとに判定時間をランダム設定",
                 bg="#f0f0f0", fg="#666", font=("Arial", 8)).pack(anchor="w")
        tk.Button(ctrl, text="ランダム建物生成", command=self.generate_random,
                  bg="#4CAF50", fg="white", relief=tk.FLAT, pady=4, cursor="hand2"
                  ).pack(fill=tk.X, pady=2)
        tk.Button(ctrl, text="クリア", command=self.clear_nodes,
                  bg="#f44336", fg="white", relief=tk.FLAT, pady=3, cursor="hand2"
                  ).pack(fill=tk.X, pady=1)

        # デポ設定セクション
        self._sep(ctrl, "デポ（拠点）")
        for val, lbl in [("center","中心に自動配置"),
                          ("random","ランダム"),
                          ("click", "クリックで指定")]:
            tk.Radiobutton(ctrl, text=lbl, variable=self.depot_mode_var,
                           value=val, bg="#f0f0f0").pack(anchor="w")
        self.depot_label = tk.Label(ctrl, text="デポ: 未設定",
                                    bg="#f0f0f0", fg="#555", font=("Arial",8), anchor="w")
        self.depot_label.pack(fill=tk.X)

        # 制約条件セクション
        self._sep(ctrl, "制約条件")
        self._row(ctrl, "判定士数 m:",        "m_var",       "4",              6)
        self._row(ctrl, "移動速度 (km/h):",   "speed_var",  "30",              6)
        self._row(ctrl, "最大稼働時間 (h):",  "maxwork_var", "8",              6)
        self._row(ctrl, "並列試行回数:",      "starts_var",  str(self.n_cpu),  6)
        self._row(ctrl, "並列数:",            "workers_var", str(self.n_cpu),  6)
        tk.Label(ctrl,
                 text="目的関数: 最大終了時間(makespan)を最小化\n"
                      "→ 最も遅い担当者が早く終わるよう割当",
                 bg="#f0f0f0", fg="#1565C0", font=("Arial",8),
                 justify=tk.LEFT).pack(anchor="w", pady=3)

        # 実行ボタン群
        self.solve_btn = tk.Button(
            ctrl, text="計画を実行", command=self.solve_tsp,
            bg="#2196F3", fg="white", relief=tk.FLAT, pady=7, cursor="hand2",
            font=("Arial", 10, "bold"))
        self.solve_btn.pack(fill=tk.X, pady=4)

        self.min_m_btn = tk.Button(
            ctrl, text="必要判定士数を計算", command=self.calc_min_m,
            bg="#00796B", fg="white", relief=tk.FLAT, pady=5, cursor="hand2",
            font=("Arial", 9, "bold"))
        self.min_m_btn.pack(fill=tk.X, pady=2)

        self.stop_btn = tk.Button(
            ctrl, text="中止", command=lambda: setattr(self,"solving",False),
            bg="#f44336", fg="white", relief=tk.FLAT, pady=3, cursor="hand2",
            state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=1)

        # 結果表示セクション
        self._sep(ctrl, "結果")
        self.stat_nodes    = self._lbl(ctrl, "建物数: 0")
        self.stat_m        = self._lbl(ctrl, "判定士数: -")
        self.stat_min_m    = self._lbl(ctrl, "最小必要判定士数: -",
                                       bold=True, color="#00796B")
        self.stat_makespan = self._lbl(ctrl, "最大終了時間: -", bold=True)
        self.stat_unassign = self._lbl(ctrl, "未割当建物: -", color="#e74c3c")
        self.stat_dist     = self._lbl(ctrl, "総移動距離: -")
        self.stat_time     = self._lbl(ctrl, "計算時間: -")
        self.stat_status   = tk.Label(ctrl, text="状態: 待機中", bg="#f0f0f0",
                                      fg="#333", anchor="w", wraplength=270,
                                      justify=tk.LEFT)
        self.stat_status.pack(fill=tk.X)

        # 判定士ごとの稼働時間・移動距離テーブル
        self.per_text = tk.Text(ctrl, height=7, font=("Courier", 8),
                                state=tk.DISABLED)
        self.per_text.pack(fill=tk.X, pady=4)

        self.progress = ttk.Progressbar(ctrl, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=4)

        self._sep(ctrl, "ベンチマーク")
        self.bench_btn = tk.Button(
            ctrl, text="ベンチマーク実行", command=self.run_benchmark,
            bg="#9C27B0", fg="white", relief=tk.FLAT, pady=4, cursor="hand2")
        self.bench_btn.pack(fill=tk.X, pady=3)

        # 右側キャンバス（matplotlib）
        right = tk.Frame(self.root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor("#ffffff")
        self.fig.patch.set_facecolor("#ffffff")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self._redraw()

    def _sep(self, parent, text):
        """セパレータ＋セクションラベルを追加するヘルパー"""
        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=5)
        tk.Label(parent, text=text, font=("Arial", 9, "bold"),
                 bg="#f0f0f0", anchor="w").pack(fill=tk.X)

    def _row(self, parent, label, attr, default, width):
        """ラベル＋入力フィールドの行を追加するヘルパー"""
        frm = tk.Frame(parent, bg="#f0f0f0")
        frm.pack(fill=tk.X, pady=2)
        tk.Label(frm, text=label, bg="#f0f0f0", width=16, anchor="w").pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        setattr(self, attr, var)
        tk.Entry(frm, textvariable=var, width=width).pack(side=tk.LEFT)

    def _lbl(self, parent, text, bold=False, color="#333"):
        """統計表示ラベルを追加するヘルパー"""
        font = ("Arial", 9, "bold") if bold else ("Arial", 9)
        lbl = tk.Label(parent, text=text, bg="#f0f0f0", fg=color,
                       anchor="w", font=font)
        lbl.pack(fill=tk.X)
        return lbl

    # ── 建物管理 ─────────────────────────────────────────────────────────────

    def generate_random(self):
        """指定棟数の建物をランダム配置し、判定時間をランダム設定する"""
        try:
            n     = int(self.n_var.get());      assert 2 <= n <= 2_000_000
            t_min = float(self.tmin_var.get()); assert t_min > 0
            t_max = float(self.tmax_var.get()); assert t_max >= t_min
        except Exception:
            messagebox.showerror("エラー", "入力値を確認してください")
            return

        xy = np.random.random((n, 2))
        self.nodes = list(map(tuple, xy.tolist()))
        # 判定時間を [t_min, t_max] 分の範囲でランダム設定（秒換算で保持）
        self.inspect_times = np.random.uniform(
            t_min * 60, t_max * 60, n).tolist()
        self.routes = []
        self.per_time = []
        self.per_dist = []
        self.unassigned = 0
        self.depot_idx = self._auto_depot()
        self.stat_nodes.config(text=f"建物数: {n:,}")
        self.depot_label.config(text=f"デポ: 拠点 #{self.depot_idx}")
        self._reset_result_stats()
        self._redraw()

    def clear_nodes(self):
        self.nodes = []
        self.inspect_times = []
        self.routes = []
        self.per_time = []
        self.per_dist = []
        self.unassigned = 0
        self.stat_nodes.config(text="建物数: 0")
        self._reset_result_stats()
        self._redraw()

    def _reset_result_stats(self):
        self.stat_m.config(text="判定士数: -")
        self.stat_min_m.config(text="最小必要判定士数: -")
        self.stat_makespan.config(text="最大終了時間: -")
        self.stat_unassign.config(text="未割当建物: -")
        self.stat_dist.config(text="総移動距離: -")
        self.stat_time.config(text="計算時間: -")
        self.stat_status.config(text="状態: 待機中")
        self._update_per_text([])

    def calc_min_m(self):
        """
        二分探索で全棟割当可能な最小判定士数を求める。

        探索範囲: [1, 建物数]
        各試行で _mtsp_worker を呼び unassigned == 0 かどうかを確認。
        O(log n) 回の試行で収束する。
        """
        if len(self.nodes) < 2:
            messagebox.showwarning("警告", "建物を2棟以上追加してください")
            return
        if self.solving:
            return
        try:
            speed    = float(self.speed_var.get());    assert speed > 0
            max_work = float(self.maxwork_var.get());  assert max_work > 0
            area_km  = float(self.area_var.get());     assert area_km > 0
        except Exception:
            messagebox.showerror("エラー", "移動速度・最大稼働時間・エリアを確認してください")
            return

        self.solving = True
        self.solve_btn.config(state=tk.DISABLED)
        self.min_m_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.stat_status.config(text="状態: 最小判定士数を探索中...")
        self.progress.start(10)

        threading.Thread(
            target=self._min_m_worker,
            args=(list(self.nodes), list(self.inspect_times),
                  speed, max_work, area_km),
            daemon=True).start()

    def _min_m_worker(self, nodes, inspect_times, speed, max_work, area_km):
        """二分探索の実体。バックグラウンドスレッドで実行される。"""
        coords = np.array(nodes)
        it     = np.array(inspect_times, dtype=np.float64)
        depot  = self._auto_depot()
        t0     = time.perf_counter()

        lo, hi   = 1, len(nodes)
        best_m   = hi
        best_res = None

        while lo <= hi:
            if not self.solving:
                break
            mid = (lo + hi) // 2
            self.root.after(0, self.stat_status.config,
                            {"text": f"状態: 探索中... m={mid} を試行"})
            res        = _parallel_mtsp(coords, it, [depot],
                                        mid, area_km, speed, max_work, 1)
            unassigned = res[5]
            if unassigned == 0:
                # m で全棟対応可能 → さらに少ない m を試す
                best_m   = mid
                best_res = res
                hi       = mid - 1
            else:
                # m では足りない → m を増やす
                lo = mid + 1

        elapsed = time.perf_counter() - t0
        self.root.after(0, self._on_min_m_done, best_m, best_res, elapsed)

    def _on_min_m_done(self, min_m, res, elapsed):
        """最小判定士数探索完了時のコールバック（メインスレッドで実行）"""
        self.solving = False
        self.solve_btn.config(state=tk.NORMAL)
        self.min_m_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress.stop()

        self.stat_min_m.config(text=f"最小必要判定士数: {min_m} 人")
        self.stat_time.config(text=f"計算時間: {elapsed:.2f} 秒")
        self.m_var.set(str(min_m))  # 判定士数入力欄に反映

        if res:
            makespan, total_dist, routes, per_time, per_dist, _ = res
            self.routes     = routes
            self.per_time   = per_time
            self.per_dist   = per_dist
            self.unassigned = 0
            self.depot_idx  = routes[0][0]
            h = int(makespan); mn = int((makespan - h) * 60)
            self.stat_makespan.config(text=f"最大終了時間: {h}h{mn:02d}m")
            self.stat_unassign.config(text="未割当建物: 0 棟 (全棟完了)",
                                      fg="#2e7d32")
            self.stat_dist.config(text=f"総移動距離: {total_dist:.1f} km")
            self.stat_m.config(text=f"判定士数: {min_m} (最小)")
            self._update_per_text(per_time, per_dist)
            self._redraw()

        self.stat_status.config(
            text=f"状態: 完了 — 最小 {min_m} 人で全棟対応可能")

    def _auto_depot(self):
        """デポ設定モードに応じてデポのインデックスを返す"""
        mode = self.depot_mode_var.get()
        if not self.nodes:
            return 0
        if mode == "center":
            # エリア重心に最も近い建物をデポに設定
            arr = np.array(self.nodes)
            c   = arr.mean(axis=0)
            return int(np.argmin(np.linalg.norm(arr - c, axis=1)))
        return random.randrange(len(self.nodes))

    def on_canvas_click(self, event):
        """キャンバスクリック: デポ指定モードならデポを変更、通常は建物を追加"""
        if event.inaxes != self.ax or self.solving:
            return
        x, y = event.xdata, event.ydata
        if x is None:
            return
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        nx = (x - xlim[0]) / (xlim[1] - xlim[0])
        ny = (y - ylim[0]) / (ylim[1] - ylim[0])

        if self.depot_mode_var.get() == "click" and self.nodes:
            # クリック位置に最も近い既存建物をデポに設定
            arr = np.array(self.nodes)
            self.depot_idx = int(
                np.argmin(np.linalg.norm(arr - [nx, ny], axis=1)))
            self.depot_label.config(
                text=f"デポ: 拠点 #{self.depot_idx} (クリック指定)")
            self.routes = []
            self._redraw()
        else:
            try:
                t_min = float(self.tmin_var.get())
                t_max = float(self.tmax_var.get())
            except Exception:
                t_min, t_max = 15, 45
            self.nodes.append((nx, ny))
            self.inspect_times.append(random.uniform(t_min * 60, t_max * 60))
            self.routes = []
            self.stat_nodes.config(text=f"建物数: {len(self.nodes):,}")
            self._redraw()

    # ── 描画 ─────────────────────────────────────────────────────────────────

    def _redraw(self):
        """
        matplotlib キャンバスを再描画する。
        fig.clear() で全要素（カラーバー含む）をリセットしてから再描画することで
        ランダム生成を繰り返してもカラーバーが重複しない。
        """
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#ffffff")
        self.ax.set_xlim(-0.02, 1.02)
        self.ax.set_ylim(-0.02, 1.02)
        self.ax.tick_params(colors="#666")
        for spine in self.ax.spines.values():
            spine.set_color("#cccccc")

        n     = len(self.nodes)
        extra = f"  未割当: {self.unassigned}" if self.unassigned else ""
        self.ax.set_title(
            f"応急危険度判定 mTSP  ({n:,} 建物{extra})",
            color="#333333", fontsize=11)

        if not self.nodes:
            self.canvas.draw_idle()
            return

        coords = np.array(self.nodes)

        # 判定士ごとのルートを色分けして描画
        if self.routes:
            for s, route in enumerate(self.routes):
                color = COLORS[s % len(COLORS)]
                rc    = coords[route]
                lw    = max(0.4, 1.4 - n / 25000)
                self.ax.plot(rc[:, 0], rc[:, 1], "-",
                             color=color, linewidth=lw, alpha=0.75, zorder=2)

        # 建物を判定時間のヒートマップで描画（黄→赤: 短い→長い）
        if self.inspect_times and len(self.inspect_times) == n:
            it   = np.array(self.inspect_times) / 60.0  # 秒→分
            size = max(2, 25 - n // 1000)
            sc   = self.ax.scatter(coords[:, 0], coords[:, 1],
                                   c=it, cmap="YlOrRd", s=size, alpha=0.75,
                                   zorder=3, vmin=it.min(), vmax=it.max())
            try:
                cb = self.fig.colorbar(sc, ax=self.ax, fraction=0.03, pad=0.01)
                cb.set_label("判定時間 (分)", color="#333333", fontsize=8)
                cb.ax.yaxis.set_tick_params(color="#666", labelcolor="#333333")
            except Exception:
                pass
        else:
            size = max(2, 25 - n // 1000)
            self.ax.scatter(coords[:, 0], coords[:, 1],
                            c="#4CAF50", s=size, alpha=0.75, zorder=3)

        # デポを金色の星マークで表示
        dep = self.depot_idx if self.nodes else 0
        if 0 <= dep < n:
            self.ax.scatter([coords[dep, 0]], [coords[dep, 1]],
                            c="#E53935", s=200, marker="*",
                            edgecolors="white", linewidths=0.8, zorder=6,
                            label="デポ")
            self.ax.legend(loc="upper right", fontsize=9,
                           facecolor="#ffffff", labelcolor="#333333",
                           edgecolor="#cccccc")

        self.canvas.draw_idle()

    def _update_per_text(self, per_time, per_dist=None):
        """判定士ごとの稼働時間・移動距離テーブルを更新する"""
        self.per_text.config(state=tk.NORMAL)
        self.per_text.delete("1.0", tk.END)
        if per_time:
            self.per_text.insert(
                tk.END, f"{'担当者':>5}  {'稼働時間(h)':>11}  {'移動距離(km)':>12}\n")
            self.per_text.insert(tk.END, "-" * 34 + "\n")
            for i, t in enumerate(per_time):
                d_str = f"{per_dist[i]:>12.2f}" if per_dist else ""
                self.per_text.insert(
                    tk.END, f"  #{i+1:2d}   {t:>11.3f}  {d_str}\n")
        self.per_text.config(state=tk.DISABLED)

    # ── ソルバー実行 ─────────────────────────────────────────────────────────

    def solve_tsp(self):
        """「計画を実行」ボタンのハンドラ。バックグラウンドスレッドで解を求める。"""
        if len(self.nodes) < 2:
            messagebox.showwarning("警告", "建物を2棟以上追加してください")
            return
        if self.solving:
            return
        try:
            m         = max(1, int(self.m_var.get()))
            speed     = float(self.speed_var.get());    assert speed > 0
            max_work  = float(self.maxwork_var.get());  assert max_work > 0
            area_km   = float(self.area_var.get());     assert area_km > 0
            n_workers = max(1, int(self.workers_var.get()))
            n_starts  = max(1, int(self.starts_var.get()))
        except Exception:
            messagebox.showerror("エラー", "入力値を確認してください")
            return

        if m > len(COLORS):
            messagebox.showerror("エラー", f"判定士数は {len(COLORS)} 以下にしてください")
            return

        # デポ候補リストを生成（並列試行回数分）
        mode = self.depot_mode_var.get()
        if mode == "click":
            depots = [self.depot_idx] * n_starts
        elif mode == "center":
            depots = [self._auto_depot()] * n_starts
        else:
            depots = random.sample(range(len(self.nodes)),
                                   min(n_starts, len(self.nodes)))

        self.solving = True
        self.solve_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.stat_status.config(text="状態: 計算中...")
        self.stat_m.config(text=f"判定士数: {m}")
        self.progress.start(10)

        threading.Thread(
            target=self._solve_worker,
            args=(list(self.nodes), list(self.inspect_times),
                  m, speed, max_work, area_km, n_workers, depots),
            daemon=True).start()

    def _solve_worker(self, nodes, inspect_times, m, speed,
                      max_work, area_km, n_workers, depots):
        """ソルバーのバックグラウンドスレッド本体"""
        coords = np.array(nodes)
        it     = np.array(inspect_times, dtype=np.float64)
        t0     = time.perf_counter()
        try:
            res     = _parallel_mtsp(coords, it, depots,
                                     m, area_km, speed, max_work, n_workers)
            elapsed = time.perf_counter() - t0
            self.root.after(0, self._on_done, res, elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            self.root.after(0, self._on_done, None, elapsed, f"エラー: {e}")

    def _on_done(self, res, elapsed, err=None):
        """ソルバー完了時のコールバック（メインスレッドで実行）"""
        self.solving = False
        self.solve_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress.stop()

        if err:
            self.stat_status.config(text=f"状態: {err}")
            return
        if res is None:
            self.stat_status.config(text="状態: 失敗")
            return

        makespan, total_dist, routes, per_time, per_dist, unassigned = res
        self.routes     = routes
        self.per_time   = per_time
        self.per_dist   = per_dist
        self.unassigned = unassigned
        self.depot_idx  = routes[0][0]

        h = int(makespan); mn = int((makespan - h) * 60)
        self.stat_makespan.config(text=f"最大終了時間: {h}h{mn:02d}m")

        # 未割当があれば赤字で警告
        color  = "#e74c3c" if unassigned > 0 else "#2e7d32"
        ua_txt = (f"未割当建物: {unassigned:,} 棟 ← 時間不足"
                  if unassigned else "未割当建物: 0 棟 (全棟完了)")
        self.stat_unassign.config(text=ua_txt, fg=color)
        self.stat_dist.config(text=f"総移動距離: {total_dist:.1f} km")
        self.stat_time.config(text=f"計算時間: {elapsed:.3f} 秒")
        self.stat_status.config(text="状態: 完了")
        self._update_per_text(per_time, per_dist)
        self._redraw()

    # ── ベンチマーク ─────────────────────────────────────────────────────────

    def run_benchmark(self):
        """建物数を段階的に増やして計算時間・makespan を計測する"""
        if self.solving:
            return
        try:
            m        = max(1, int(self.m_var.get()))
            speed    = float(self.speed_var.get())
            max_work = float(self.maxwork_var.get())
            area_km  = float(self.area_var.get())
        except Exception:
            m, speed, max_work, area_km = 4, 30, 8, 10
        self.bench_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        threading.Thread(target=self._bench_worker,
                         args=(m, speed, max_work, area_km),
                         daemon=True).start()

    def _bench_worker(self, m, speed, max_work, area_km):
        sizes     = [100, 500, 1000, 5000, 10000, 50000, 100000]
        n_workers = self.n_cpu
        results   = []
        for n in sizes:
            self.root.after(0, self.stat_status.config,
                            {"text": f"ベンチマーク中: n={n:,}"})
            coords = np.random.random((n, 2)).astype(np.float64)
            it     = np.random.uniform(15*60, 45*60, n).astype(np.float64)
            t0     = time.perf_counter()
            res    = _parallel_mtsp(coords, it, [0],
                                    m, area_km, speed, max_work, 1)
            elapsed = time.perf_counter() - t0
            makespan, total_dist, _, _, _, unassigned = res
            results.append((n, elapsed, makespan, unassigned))
        self.root.after(0, self._show_bench, results, m, max_work)

    def _show_bench(self, results, m, max_work):
        self.progress.stop()
        self.bench_btn.config(state=tk.NORMAL)
        self.stat_status.config(text="状態: ベンチマーク完了")

        win = tk.Toplevel(self.root)
        win.title(f"ベンチマーク結果 (m={m} 人, 最大{max_work}h)")
        win.geometry("700x520")

        fig = Figure(figsize=(7, 5))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ns       = [r[0] for r in results]
        times    = [r[1] for r in results]
        makespan = [r[2] for r in results]
        unassign = [r[3] for r in results]

        ax1.loglog(ns, times, "o-", color="#2196F3")
        ax1.set_xlabel("建物数")
        ax1.set_ylabel("計算時間 (秒, log)")
        ax1.set_title(f"建物数 vs 計算時間 (m={m}人)")
        ax1.grid(True, alpha=0.3, which="both")

        ax2_r = ax2.twinx()
        ax2.semilogx(ns, makespan, "s-",  color="#e74c3c", label="最大終了時間 (h)")
        ax2_r.semilogx(ns, unassign, "^--", color="#9b59b6", label="未割当棟数")
        ax2.set_xlabel("建物数")
        ax2.set_ylabel("最大終了時間 (h)", color="#e74c3c")
        ax2_r.set_ylabel("未割当棟数",     color="#9b59b6")
        ax2.set_title("Makespan と未割当建物数")
        ax2.grid(True, alpha=0.3)
        lines1, lbls1 = ax2.get_legend_handles_labels()
        lines2, lbls2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1+lines2, lbls1+lbls2, fontsize=8)

        fig.tight_layout(pad=2)
        c = FigureCanvasTkAgg(fig, master=win)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        txt = tk.Text(win, height=7, font=("Courier", 9))
        txt.pack(fill=tk.X, padx=8, pady=4)
        txt.insert(tk.END,
            f"{'建物数':>8}  {'計算時間(秒)':>12}  {'Makespan(h)':>12}  {'未割当':>8}\n")
        txt.insert(tk.END, "-" * 48 + "\n")
        for n, t, ms, ua in results:
            txt.insert(tk.END,
                f"{n:>8,}  {t:>12.3f}  {ms:>12.3f}  {ua:>8,}\n")
        txt.config(state=tk.DISABLED)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = TSPApp(root)
    root.mainloop()
