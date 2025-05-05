import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import json
import random
import os
from datetime import datetime
import configparser
from itertools import combinations, permutations
import traceback
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DeepRacingPredictor:
    def __init__(self, input_dim=10, seq_length=7):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.seq_length = seq_length
        self.input_dim = input_dim

    def create_model(self):
        self.model = Sequential([
            LSTM(128, input_shape=(self.seq_length, self.input_dim),  # 과거 7경기 데이터 기반
                 Dropout(0.3),
                 Dense(64, activation='relu'),
                 Dense(3, activation='softmax')  # 1,2,3위 확률 예측
        ])
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def preprocess_data(self, raw_data):
        """경주 데이터 전처리"""
        # 데이터 정규화
        scaled = self.scaler.fit_transform(raw_data)

        # 시퀀스 데이터 생성
        sequences = []
        labels = []
        for i in range(len(scaled) - self.seq_length - 3):  # 최근 3개월(90일) 데이터 활용
            seq = scaled[i:i + self.seq_length]
            label = scaled[i + self.seq_length:i + self.seq_length + 3, 0]  # 다음 경기 1,2,3위 정보
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

    def train(self, X_train, y_train, epochs=100):
        early_stop = EarlyStopping(monitor='val_loss', patience=10)
        checkpoint = ModelCheckpoint('best_racing_model.h5',
                                     save_best_only=True,
                                     monitor='val_accuracy')
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=32,
            callbacks=[early_stop, checkpoint]
        )
        return history


class kra_bettingApp:
    def __init__(self, root):
        # ... (기존 초기화 코드)

        # 딥러닝 모듈 초기화
        self.predictor = DeepRacingPredictor()
        self.model_loaded = False

        # 예측 결과 표시 프레임 추가
        self.create_prediction_panel()

    def create_prediction_panel(self):
        """딥러닝 예측 결과 표시 패널"""
        pred_frame = ttk.LabelFrame(self.root, text="AI 예측 결과 (딥러닝)")
        pred_frame.pack(pady=10, padx=10, fill="x")

        # 예측 정보 표시 테이블
        columns = [
            ("말 번호", 70),
            ("예측 순위", 100),
            ("승률", 100),
            ("신뢰도", 100),
            ("역대 패턴", 150)
        ]

        self.pred_tree = ttk.Treeview(
            pred_frame,
            columns=[col[0] for col in columns],
            show="headings",
            height=5
        )

        for col, width in columns:
            self.pred_tree.heading(col, text=col)
            self.pred_tree.column(col, width=width, anchor="center")

        self.pred_tree.pack(fill="x", padx=5, pady=5)

        # 모델 제어 버튼
        ctrl_frame = ttk.Frame(pred_frame)
        ctrl_frame.pack(pady=5)

        ttk.Button(ctrl_frame, text="모델 학습 시작",
                   command=self.start_training).grid(row=0, column=0, padx=5)
        ttk.Button(ctrl_frame, text="실시간 예측",
                   command=self.run_prediction).grid(row=0, column=1, padx=5)
        ttk.Button(ctrl_frame, text="모델 불러오기",
                   command=self.load_model).grid(row=0, column=2, padx=5)

    def load_historical_data(self):
        """과거 경기 데이터 불러오기"""
        try:
            df = pd.read_csv('racing_data.csv', encoding='euc-kr')
            features = df[[
                'total_bets', 'refund_rate', 'win_rate',
                'position_rate', 'last_3_ranks', 'popularity'
            ]]
            targets = df[['result_1st', 'result_2nd', 'result_3rd']]
            return features.values, targets.values
        except Exception as e:
            messagebox.showerror("오류", f"데이터 불러오기 실패: {str(e)}")
            return None, None

    def start_training(self):
        """모델 학습 시작"""
        X, y = self.load_historical_data()
        if X is None or y is None:
            return

        self.predictor.create_model()
        X_seq, y_seq = self.predictor.preprocess_data(X)

        # 데이터 분할
        split = int(0.8 * len(X_seq))
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        # 모델 학습
        history = self.predictor.train(X_train, y_train)
        self.plot_training_history(history)
        messagebox.showinfo("완료", "모델 학습이 완료되었습니다")
        self.model_loaded = True

    def run_prediction(self):
        """실시간 예측 실행"""
        if not self.model_loaded:
            messagebox.showwarning("경고", "먼저 모델을 학습시키거나 불러와주세요")
            return

        # 현재 입력 데이터 추출
        current_data = self.prepare_prediction_input()

        # 모델 예측
        prediction = self.predictor.model.predict(current_data)

        # 예측 결과 처리
        sorted_indices = np.argsort(prediction[0])[::-1]
        horses = [self.calculated_data[i][0] for i in sorted_indices]
        probs = prediction[0][sorted_indices]

        # 트리뷰 업데이트
        self.pred_tree.delete(*self.pred_tree.get_children())
        for idx, (horse, prob) in enumerate(zip(horses, probs)):
            self.pred_tree.insert("", "end", values=(
                horse,
                f"{idx + 1}위 예측",
                f"{prob * 100:.2f}%",
                self.calculate_confidence(prob),
                self.get_pattern_analysis(horse)
            ))

    def prepare_prediction_input(self):
        """현재 입력값을 모델 입력 형태로 변환"""
        # 현재 베팅 데이터 수집
        features = []
        for horse in self.calculated_data:
            features.append([
                horse[1],  # 베팅금액
                self.refund_rate.get(),  # 환수율
                horse[2],  # 배당률
                horse[4],  # 베팅 비율
                horse[6]  # 안정성 점수
            ])

        # 데이터 스케일링 및 시퀀스 형식으로 변환
        scaled = self.predictor.scaler.transform(features)
        seq_data = scaled[-self.predictor.seq_length:]  # 최근 7개 데이터 사용
        return np.array([seq_data])  # 배치 차원 추가

    def calculate_confidence(self, probability):
        """신뢰도 점수 계산"""
        if probability > 0.7:
            return "⭐⭐⭐⭐⭐"
        elif probability > 0.5:
            return "⭐⭐⭐⭐"
        elif probability > 0.3:
            return "⭐⭐⭐"
        else:
            return "⭐⭐"

    def get_pattern_analysis(self, horse_num):
        """역대 패턴 분석"""
        patterns = {
            1: "초반 스퍼트 특화",
            2: "종반 반동 강함",
            3: "안정적 선두유지",
            4: "중간 지속형",
            5: "역전 주자"
        }
        return patterns.get(horse_num % 5 + 1, "패턴 없음")

    def load_model(self):
        """저장된 모델 불러오기"""
        try:
            self.predictor.model = load_model('best_racing_model.h5')
            self.model_loaded = True
            messagebox.showinfo("성공", "최적 모델 불러오기 완료")
        except Exception as e:
            messagebox.showerror("오류", f"모델 불러오기 실패: {str(e)}")

    def plot_training_history(self, history):
        """학습 과정 시각화"""
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Training Progress')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()


class kra_bettingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KRA 경마 베팅 시스템 v9.7 (3복승식 추가)")
        self.root.geometry("1600x1200")

        # 설정 파일 초기화
        self.config = configparser.ConfigParser()
        self.config_file = "config.ini"
        self.last_save_folder = os.path.expanduser("~/Desktop")

        # 설정 파일 로드
        self.load_config()

        # 변수 초기화
        self.title_var = tk.StringVar(value=f"{datetime.now().strftime('%Y%m%d_%H%M')}")
        self.total_pool = tk.StringVar(value="10000000")
        self.refund_rate = tk.DoubleVar(value=75.0)
        self.betting_data = {i: 0.0 for i in range(1, 16)}
        self.current_file_path = None
        self.current_edit_entry = None
        self.sort_order = {"amount": "desc"}
        self.calculated_data = []
        self.strategy_trees = []
        self.amount_rank_data = []
        self.high_risk_data = []
        self.final_rankings = [tk.StringVar() for _ in range(3)]

        # GUI 컴포넌트 생성
        self.create_input_section()
        self.create_betting_tables()
        self.create_result_tables()
        self.create_buttons()
        self.create_strategy_tabs()
        self.create_final_ranking_section()
        self.create_ranking_info_section()  # 가장 마지막으로 이동
        self.create_high_risk_section()


    def load_config(self):
        """설정 파일 로드"""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
            if 'DEFAULT' in self.config and 'last_save_folder' in self.config['DEFAULT']:
                if os.path.isdir(self.config['DEFAULT']['last_save_folder']):
                    self.last_save_folder = self.config['DEFAULT']['last_save_folder']

    def save_config(self):
        """설정 파일 저장"""
        self.config['DEFAULT'] = {
            'last_save_folder': self.last_save_folder
        }
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

    def create_input_section(self):
        """입력 섹션 생성 - 엔터 키 이동 기능 추가"""
        input_frame = ttk.Frame(self.root)
        input_frame.pack(pady=10, padx=10, fill="x")

        # 제목 입력
        title_frame = ttk.LabelFrame(input_frame, text="제목 입력")
        title_frame.pack(side="left", padx=5, fill="x", expand=True)
        ttk.Label(title_frame, text="제목:").grid(row=0, column=0)
        self.title_entry = ttk.Entry(title_frame, textvariable=self.title_var)
        self.title_entry.grid(row=0, column=1)
        self.title_entry.bind("<Return>", lambda e: self.total_entry.focus())

        # 총 베팅 금액 입력
        pool_frame = ttk.LabelFrame(input_frame, text="총 베팅 금액 입력 (소수점 가능)")
        pool_frame.pack(side="left", padx=5, fill="x", expand=True)
        ttk.Label(pool_frame, text="총 금액 (원):").grid(row=0, column=0)
        self.total_entry = ttk.Entry(pool_frame, textvariable=self.total_pool)
        self.total_entry.grid(row=0, column=1)
        self.total_entry.bind("<Return>", lambda e: self.refund_entry.focus())

        # 환수율 입력
        refund_frame = ttk.LabelFrame(input_frame, text="환수율 설정 (%)")
        refund_frame.pack(side="left", padx=5, fill="x", expand=True)
        ttk.Label(refund_frame, text="환수율 (%):").grid(row=0, column=0)
        self.refund_entry = ttk.Entry(refund_frame, textvariable=self.refund_rate)
        self.refund_entry.grid(row=0, column=1)
        self.refund_entry.bind("<Return>", lambda e: self.focus_first_betting_row())
        ttk.Label(refund_frame, text="※ 한국마사회 기준: 75%").grid(row=0, column=2, padx=5)

    def focus_first_betting_row(self):
        """첫 번째 베팅 행으로 포커스 이동"""
        if self.tree.get_children():
            first_item = self.tree.get_children()[0]
            self.tree.focus(first_item)
            self.tree.selection_set(first_item)
            self.edit_selected_row()

    def edit_selected_row(self):
        """선택된 행 편집 모드 진입 - 엔터 키 이동 개선"""
        selected = self.tree.selection()
        if not selected:
            return

        if not self.tree.exists(selected[0]):
            return

        # bbox()가 값을 반환하지 않으면 처리 중단
        bbox_values = self.tree.bbox(selected[0], "#2")
        if not bbox_values:
            return

        x, y, width, height = bbox_values
        current_value = self.tree.item(selected[0], "values")[1]

        # 기존 편집 엔트리가 있으면 제거
        if self.current_edit_entry:
            self.current_edit_entry.destroy()
            self.current_edit_entry = None

        x, y, width, height = self.tree.bbox(selected[0], "#2")
        current_value = self.tree.item(selected[0], "values")[1]

        self.current_edit_entry = ttk.Entry(self.tree)
        self.current_edit_entry.place(x=x, y=y, width=width, height=height)
        self.current_edit_entry.insert(0, current_value)
        self.current_edit_entry.select_range(0, tk.END)
        self.current_edit_entry.focus()

        # 엔터 키 바인딩 (저장 후 다음 행으로 이동)
        self.current_edit_entry.bind("<Return>",
                                    lambda e: [self.save_edit(selected[0]),
                                               self.focus_next_row_or_calculate()])

        # 포커스 아웃 시 저장 (단, 위젯이 아직 존재하는 경우에만)
        self.current_edit_entry.bind("<FocusOut>",
                                    lambda e: self.save_edit(selected[0]) if self.current_edit_entry else None)

    def update_sort_indicator(self, tree, column, direction):
        """정렬 방향 표시 업데이트"""
        # 모든 컬럼의 heading에서 정렬 표시 제거
        for col in tree["columns"]:
            heading_text = tree.heading(col)["text"]
            if heading_text.endswith(" ↑") or heading_text.endswith(" ↓"):
                tree.heading(col, text=heading_text[:-2])

        # 현재 컬럼에 정렬 방향 표시 추가
        current_text = tree.heading(column)["text"]
        arrow = " ↑" if direction == "asc" else " ↓"
        tree.heading(column, text=current_text + arrow)

    def focus_next_row_or_calculate(self):
        """다음 행으로 이동 또는 계산 실행"""
        current_item = self.tree.focus()
        next_item = self.tree.next(current_item)

        if next_item and self.tree.exists(next_item):
            self.tree.focus(next_item)
            self.tree.selection_set(next_item)
            self.edit_selected_row()
        else:
            self.calculate_odds()
            self.total_entry.focus()

    def create_final_ranking_section(self):
        """최종 순위 입력 섹션 생성 (엔터 키 이동 기능 추가)"""
        rank_frame = ttk.LabelFrame(self.root, text="최종 순위 입력 (1, 2, 3위) - Enter로 이동")
        rank_frame.pack(pady=10, padx=10, fill="x")

        # 순위 입력 필드 생성
        self.rank_entries = []
        for i in range(3):
            ttk.Label(rank_frame, text=f"{i + 1}위:").grid(row=0, column=i * 2, padx=5)
            entry = ttk.Entry(rank_frame, textvariable=self.final_rankings[i], width=5)
            entry.grid(row=0, column=i * 2 + 1, padx=5)
            self.rank_entries.append(entry)

        # 엔터 키 바인딩 설정
        for i in range(len(self.rank_entries)):
            if i < len(self.rank_entries) - 1:
                # 다음 순위 필드로 이동
                self.rank_entries[i].bind("<Return>",
                                          lambda e, idx=i: self.rank_entries[idx + 1].focus())
            else:
                # 마지막 순위 필드에서는 계산 실행
                self.rank_entries[i].bind("<Return>",
                                          lambda e: [self.calculate_odds(),
                                                     self.total_entry.focus()])

    def create_ranking_info_section(self):
        """최종 순위 상세 정보 표시 섹션 생성 (맨 아래 배치)"""
        self.ranking_info_frame = ttk.LabelFrame(self.root, text="최종 순위 상세 정보")
        self.ranking_info_frame.pack(pady=10, padx=10, fill="x")  # side 추가

        columns = [
            ("순위", 70), ("번호", 70), ("베팅금액", 120),
            ("배당률", 100), ("안정성", 100), ("비율", 80), ("종합점수", 100)
        ]

        self.ranking_tree = ttk.Treeview(
            self.ranking_info_frame,
            columns=[col[0] for col in columns],
            show="headings",
            height=3
        )

        for col, width in columns:
            self.ranking_tree.heading(col, text=col)
            self.ranking_tree.column(col, width=width, anchor="center")

        self.ranking_tree.pack(fill="x", padx=5, pady=5)

    def update_ranking_info(self):
        """최종 순위 정보 업데이트"""
        self.ranking_tree.delete(*self.ranking_tree.get_children())

        for idx, rank_var in enumerate(self.final_rankings, 1):
            horse_num = rank_var.get()
            if not horse_num:
                continue

            try:
                horse_num = int(horse_num)
                horse_info = next((h for h in self.calculated_data if h[0] == horse_num), None)

                if horse_info:
                    num, bet, odds, stars, ratio, score, stability = horse_info
                    self.ranking_tree.insert("", "end", values=(
                        f"{idx}위",
                        num,
                        f"{bet:,.2f}배",
                        f"{odds:.2f}원",
                        stars,
                        f"{ratio:.1f}%",
                        f"{score:.2f}"
                    ))
                else:
                    self.ranking_tree.insert("", "end", values=(
                        f"{idx}위",
                        horse_num,
                        "정보 없음",
                        "-",
                        "-",
                        "-",
                        "-"
                    ))
            except ValueError:
                self.ranking_tree.insert("", "end", values=(
                    f"{idx}위",
                    "잘못된 입력",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-"
                ))

    def create_betting_tables(self):
        """베팅 테이블 생성 (디자인 조정: 번호 입력 테이블 확장)"""
        table_frame = ttk.Frame(self.root)
        table_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # 베팅 입력 테이블 (너비 확장)
        input_frame = ttk.LabelFrame(table_frame, text="번호별 베팅 금액 입력 (1~15번)")
        input_frame.pack(side="left", fill="both", expand=True, padx=5)

        # 컬럼 너비 조정 (번호: 60, 베팅금액: 200, 배당률: 120, 안정성: 120)
        self.tree = ttk.Treeview(
            input_frame,
            columns=("번호", "베팅금액", "배당률", "안정성 점수"),
            show="headings",
            height=15  # 높이도 조정하여 전체 번호 표시
        )
        self.tree.heading("번호", text="번호")
        self.tree.heading("베팅금액", text="베팅 금액 (배)")
        self.tree.heading("배당률", text="배당률 (원)")
        self.tree.heading("안정성 점수", text="안정성 (1~5★)")

        # 컬럼 너비 확장
        self.tree.column("번호", width=60, anchor="center", stretch=tk.YES)
        self.tree.column("베팅금액", width=200, anchor="e", stretch=tk.YES)
        self.tree.column("배당률", width=120, anchor="e", stretch=tk.YES)
        self.tree.column("안정성 점수", width=120, anchor="center", stretch=tk.YES)

        # 스크롤바 추가
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

        # 초기 데이터
        for num in range(1, 16):
            self.tree.insert("", "end", values=(num, "0.00", 0.0, "★☆☆☆☆"))

        self.tree.bind("<Double-1>", self.edit_betting)
        self.tree.bind("<Return>", lambda e: self.focus_next_row())

        # 베팅 금액 순위 테이블 (너비 축소)
        rank_frame = ttk.LabelFrame(table_frame, text="🟢 베팅 금액 순위 (클릭하여 정렬 변경)")
        rank_frame.pack(side="left", fill="both", expand=False, padx=5, ipadx=5)  # expand=False로 변경

        self.amount_rank_tree = ttk.Treeview(
            rank_frame,
            columns=("순위", "번호", "베팅금액", "비율"),
            show="headings",
            height=15
        )

        # 컬럼 너비 조정 (순위: 60, 번호: 60, 베팅금액: 150, 비율: 80)
        columns = {
            "순위": {"width": 60, "anchor": "center", "sort": False},
            "번호": {"width": 60, "anchor": "center", "sort": False},
            "베팅금액": {"width": 150, "anchor": "e", "sort": True},
            "비율": {"width": 80, "anchor": "center", "sort": True}
        }

        for col, config in columns.items():
            self.amount_rank_tree.heading(
                col,
                text=col,
                command=lambda c=col: self.sort_amount_rank(c) if config["sort"] else None
            )
            self.amount_rank_tree.column(col, width=config["width"], anchor=config["anchor"])

        # 스크롤바 추가
        scrollbar_rank = ttk.Scrollbar(rank_frame, orient="vertical", command=self.amount_rank_tree.yview)
        self.amount_rank_tree.configure(yscrollcommand=scrollbar_rank.set)
        scrollbar_rank.pack(side="right", fill="y")
        self.amount_rank_tree.pack(fill="both", expand=True)

    def create_high_risk_section(self):
        """고위험 배팅 전용 탭 생성 (4복승식 및 4마리 순위없는 조합 추가)"""
        tab_frame = ttk.Frame(self.root)
        tab_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.high_risk_notebook = ttk.Notebook(tab_frame)

        # 고위험 배팅 테이블 생성 함수
        def create_high_risk_tree(parent, columns, col_widths, sortable_columns):
            tree = ttk.Treeview(
                parent,
                columns=columns,
                show="headings",
                height=10
            )

            # 컬럼 설정
            for col in columns:
                if col in sortable_columns:
                    tree.heading(col, text=col,
                                 command=lambda c=col: self.sort_high_risk_table(tree, c))
                else:
                    tree.heading(col, text=col)

                tree.column(col, width=col_widths.get(col, 100), anchor="center")

            scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            tree.pack(fill="both", expand=True)
            return tree

        # 1. 복승식(Exacta) 탭
        exacta_frame = ttk.Frame(self.high_risk_notebook)
        self.exacta_tree = create_high_risk_tree(
            exacta_frame,
            columns=["No", "조합", "1등말", "2등말", "배당률", "승률", "예상수익"],
            col_widths={"No": 50, "조합": 100, "1등말": 80, "2등말": 80,
                        "배당률": 100, "승률": 80, "예상수익": 120},
            sortable_columns=["배당률", "승률", "예상수익"]
        )
        self.high_risk_notebook.add(exacta_frame, text="복승식(Exacta)")

        # 2. 쌍승식(Quinella) 탭
        quinella_frame = ttk.Frame(self.high_risk_notebook)
        self.quinella_tree = create_high_risk_tree(
            quinella_frame,
            columns=["No", "조합", "말1", "말2", "배당률", "승률", "예상수익"],
            col_widths={"No": 50, "조합": 100, "말1": 80, "말2": 80,
                        "배당률": 100, "승률": 80, "예상수익": 120},
            sortable_columns=["배당률", "승률", "예상수익"]
        )
        self.high_risk_notebook.add(quinella_frame, text="쌍승식(Quinella)")

        # 3. 3복승식(Trifecta) 탭
        trifecta_frame = ttk.Frame(self.high_risk_notebook)
        self.trifecta_tree = create_high_risk_tree(
            trifecta_frame,
            columns=["No", "조합", "1등", "2등", "3등", "배당률", "승률", "예상수익"],
            col_widths={"No": 50, "조합": 120, "1등": 70, "2등": 70, "3등": 70,
                        "배당률": 100, "승률": 80, "예상수익": 120},
            sortable_columns=["배당률", "승률", "예상수익"]
        )
        self.high_risk_notebook.add(trifecta_frame, text="3복승식(Trifecta)")

        # 4. 트리펙타(Triple) 탭
        triple_frame = ttk.Frame(self.high_risk_notebook)
        self.triple_tree = create_high_risk_tree(
            triple_frame,
            columns=["No", "조합", "말1", "말2", "말3", "배당률", "승률", "예상수익"],
            col_widths={"No": 50, "조합": 100, "말1": 70, "말2": 70, "말3": 70,
                        "배당률": 100, "승률": 80, "예상수익": 120},
            sortable_columns=["배당률", "승률", "예상수익"]
        )
        self.high_risk_notebook.add(triple_frame, text="트리펙타(Triple)")

        # 5. 4복승식(Quadfecta) 탭 추가 (순위O)
        quadfecta_frame = ttk.Frame(self.high_risk_notebook)
        self.quadfecta_tree = create_high_risk_tree(
            quadfecta_frame,
            columns=["No", "조합", "1등", "2등", "3등", "4등", "배당률", "승률", "예상수익"],
            col_widths={"No": 50, "조합": 140, "1등": 60, "2등": 60, "3등": 60, "4등": 60,
                        "배당률": 100, "승률": 80, "예상수익": 120},
            sortable_columns=["배당률", "승률", "예상수익"]
        )
        self.high_risk_notebook.add(quadfecta_frame, text="4복승식(Quadfecta)")

        # 6. 4마리 순위없는 조합 탭 추가 (순위X)
        quadcombo_frame = ttk.Frame(self.high_risk_notebook)
        self.quadcombo_tree = create_high_risk_tree(
            quadcombo_frame,
            columns=["No", "조합", "말1", "말2", "말3", "말4", "배당률", "승률", "예상수익"],
            col_widths={"No": 50, "조합": 140, "말1": 60, "말2": 60, "말3": 60, "말4": 60,
                        "배당률": 100, "승률": 80, "예상수익": 120},
            sortable_columns=["배당률", "승률", "예상수익"]
        )
        self.high_risk_notebook.add(quadcombo_frame, text="4마리 순위없는 조합")

        self.high_risk_sort_order = {
            "exacta": {},
            "quinella": {},
            "trifecta": {},
            "triple": {},
            "quadfecta": {},
            "quadcombo": {}
        }
        self.high_risk_notebook.pack(fill="both", expand=True)

    def generate_high_risk_combinations(self):
        """고위험 배팅 조합 생성 및 시뮬레이션 (4복승식 및 4마리 순위없는 조합 추가)"""
        if not self.calculated_data:
            messagebox.showwarning("경고", "먼저 기본 계산을 실행해주세요 (F5)")
            return

        try:
            # 활성화된 말 필터링 (베팅금액 > 0)
            active_horses = [h for h in self.calculated_data if h[1] > 0]
            if len(active_horses) < 3:
                raise ValueError("최소 3개 말에 베팅해야 고위험 배팅이 가능합니다")

            # 말별 기본 승률 계산 (베팅 금액 비율 기반)
            total_bet = sum(h[1] for h in active_horses)
            horse_win_probs = {h[0]: (h[1] / total_bet) * (h[6] / 5) for h in active_horses}

            # 상위 7개 말 사용으로 변경 (기존 5 → 7)
            top_horses = sorted(active_horses, key=lambda x: x[5], reverse=True)[:7]

            # 1. 복승식(Exacta) 조합 생성
            self.exacta_tree.delete(*self.exacta_tree.get_children())
            exacta_combos = list(permutations(top_horses, 2))

            # 3. 3복승식(Trifecta) 조합 생성 (상위 7개 사용)
            self.trifecta_tree.delete(*self.trifecta_tree.get_children())
            trifecta_combos = list(permutations(top_horses, 3))

            # 1. 복승식(Exacta) 조합 생성
            self.exacta_tree.delete(*self.exacta_tree.get_children())
            exacta_combos = list(permutations(top_horses, 2))
            for idx, (first, second) in enumerate(exacta_combos, 1):
                win_prob = horse_win_probs[first[0]] * horse_win_probs[second[0]] / (1 - horse_win_probs[first[0]])
                odds = first[2] * second[2] * (self.refund_rate.get() / 100)
                expected_return = (odds * win_prob - 1) * 100

                self.exacta_tree.insert("", "end", values=(
                    idx,
                    f"{first[0]}→{second[0]}",
                    first[0],
                    second[0],
                    f"{odds:.1f}배",
                    f"{win_prob:.4%}",
                    f"{expected_return:.1f}%"
                ))

            # 2. 쌍승식(Quinella) 조합 생성
            self.quinella_tree.delete(*self.quinella_tree.get_children())
            quinella_combos = list(combinations(top_horses, 2))
            for idx, (a, b) in enumerate(quinella_combos, 1):
                win_prob = (horse_win_probs[a[0]] * horse_win_probs[b[0]]) * 2  # 순서 무관하므로 2배
                odds = (a[2] + b[2]) / 2 * (self.refund_rate.get() / 100)
                expected_return = (odds * win_prob - 1) * 100

                self.quinella_tree.insert("", "end", values=(
                    idx,
                    f"{a[0]}+{b[0]}",
                    a[0],
                    b[0],
                    f"{odds:.1f}배",
                    f"{win_prob:.4%}",
                    f"{expected_return:.1f}%"
                ))

            # 3. 3복승식(Trifecta) 조합 생성
            self.trifecta_tree.delete(*self.trifecta_tree.get_children())
            trifecta_combos = list(permutations(top_horses, 3))
            for idx, (first, second, third) in enumerate(trifecta_combos, 1):
                # 조건부 확률 계산
                win_prob = (horse_win_probs[first[0]] *
                            (horse_win_probs[second[0]] / (1 - horse_win_probs[first[0]])) *
                            (horse_win_probs[third[0]] / (1 - horse_win_probs[first[0]] - horse_win_probs[second[0]])))

                # 안정성 점수 반영
                stability_factor = (first[6] + second[6] + third[6]) / 15
                win_prob *= stability_factor * 0.5  # 추가 보정 계수

                odds = first[2] * second[2] * third[2] * (self.refund_rate.get() / 100)
                expected_return = (odds * win_prob - 1) * 100

                self.trifecta_tree.insert("", "end", values=(
                    idx,
                    f"{first[0]}→{second[0]}→{third[0]}",
                    first[0],
                    second[0],
                    third[0],
                    f"{odds:.1f}배",
                    f"{win_prob:.6%}",
                    f"{expected_return:.2f}%"
                ))

            # 4. 트리펙타(Triple) 조합 생성
            self.triple_tree.delete(*self.triple_tree.get_children())
            triple_combos = list(combinations(top_horses, 3))
            for idx, (a, b, c) in enumerate(triple_combos, 1):
                # 조합 확률 (순서 무관)
                win_prob = horse_win_probs[a[0]] * horse_win_probs[b[0]] * horse_win_probs[c[0]] * 6  # 3! = 6

                # 안정성 점수 반영
                stability_factor = (a[6] + b[6] + c[6]) / 15
                win_prob *= stability_factor * 0.3  # 추가 보정 계수

                odds = (a[2] + b[2] + c[2]) / 3 * (self.refund_rate.get() / 100)
                expected_return = (odds * win_prob - 1) * 100

                self.triple_tree.insert("", "end", values=(
                    idx,
                    f"{a[0]}+{b[0]}+{c[0]}",
                    a[0],
                    b[0],
                    c[0],
                    f"{odds:.1f}배",
                    f"{win_prob:.4%}",
                    f"{expected_return:.1f}%"
                ))

            # 6. 4복승식(Quadfecta) 조합 생성 (순위O)
            self.quadfecta_tree.delete(*self.quadfecta_tree.get_children())
            quadfecta_combos = list(permutations(top_horses, 4))
            for idx, (first, second, third, fourth) in enumerate(quadfecta_combos, 1):
                # 확률 계산 (조건부 확률 적용)
                win_prob = (
                    horse_win_probs[first[0]] *
                    (horse_win_probs[second[0]] / (1 - horse_win_probs[first[0]])) *
                    (horse_win_probs[third[0]] / (1 - horse_win_probs[first[0]] - horse_win_probs[second[0]])) *
                    (horse_win_probs[fourth[0]] / (1 - horse_win_probs[first[0]] - horse_win_probs[second[0]] - horse_win_probs[third[0]]))
                )

                # 안정성 점수 반영
                stability_factor = (first[6] + second[6] + third[6] + fourth[6]) / 20
                win_prob *= stability_factor * 0.3  # 추가 보정 계수

                # 배당률 계산
                odds = first[2] * second[2] * third[2] * fourth[2] * (self.refund_rate.get() / 100)
                expected_return = (odds * win_prob - 1) * 100

                self.quadfecta_tree.insert("", "end", values=(
                    idx,
                    f"{first[0]}→{second[0]}→{third[0]}→{fourth[0]}",
                    first[0],
                    second[0],
                    third[0],
                    fourth[0],
                    f"{odds:.1f}배",
                    f"{win_prob:.8%}",
                    f"{expected_return:.2f}%"
                ))

            # 6. 4마리 순위없는 조합 생성 (순위X) - 개선된 버전
            self.quadcombo_tree.delete(*self.quadcombo_tree.get_children())
            quadcombo_combos = list(combinations(top_horses, 4))
            total_combinations = len(quadcombo_combos)

            for idx, (a, b, c, d) in enumerate(quadcombo_combos, 1):
                # 1. 개별 말의 가중치 계산 (베팅금액 + 안정성 점수 반영)
                weight_a = a[1] * (a[6] / 5)  # 베팅금액 * (안정성점수/5)
                weight_b = b[1] * (b[6] / 5)
                weight_c = c[1] * (c[6] / 5)
                weight_d = d[1] * (d[6] / 5)

                total_weight = weight_a + weight_b + weight_c + weight_d

                # 2. 조합의 기본 승률 계산 (가중치 기반)
                base_prob = (weight_a * weight_b * weight_c * weight_d) / (total_weight ** 4)

                # 3. 순열 개수(24)와 조정 계수 적용
                win_prob = base_prob * 24 * 0.8  # 0.8은 보정 계수 (상관관계 고려)

                # 4. 안정성 추가 반영 (4마리 평균 안정성)
                avg_stability = (a[6] + b[6] + c[6] + d[6]) / 4
                stability_factor = 0.7 + (avg_stability / 10)  # 0.8~1.2 범위
                win_prob *= stability_factor

                # 5. 배당률 계산 (복승식 특성 반영)
                combo_odds = (a[2] * b[2] * c[2] * d[2]) ** (1 / 4)  # 기하평균
                refund_adjusted = combo_odds * (self.refund_rate.get() / 100)
                odds = refund_adjusted * (1 - (1 / total_combinations))  # 조합 수 반영

                # 6. 예상수익 계산
                expected_return = (odds * win_prob - 1) * 100

                self.quadcombo_tree.insert("", "end", values=(
                    idx,
                    f"{a[0]}+{b[0]}+{c[0]}+{d[0]}",
                    a[0], b[0], c[0], d[0],
                    f"{odds:.1f}배",
                    f"{win_prob:.6%}",
                    f"{expected_return:.1f}%"
                ))

            messagebox.showinfo("완료",
                                f"고위험 배팅 조합 생성 완료:\n"
                                f"- 복승식: {len(exacta_combos)}개\n"
                                f"- 쌍승식: {len(quinella_combos)}개\n"
                                f"- 3복승식: {len(trifecta_combos)}개\n"
                                f"- 트리펙타: {len(triple_combos)}개\n"
                                f"- 4복승식: {len(quadfecta_combos)}개\n"
                                f"- 4마리 조합: {len(quadcombo_combos)}개")

        except Exception as e:
            messagebox.showerror("오류", f"고위험 배팅 생성 실패:\n{str(e)}")

    def sort_high_risk_table(self, tree, column):
        """고위험 배팅 테이블 정렬 함수 개선"""
        tab_names = {
            self.exacta_tree: "exacta",
            self.quinella_tree: "quinella",
            self.trifecta_tree: "trifecta",
            self.triple_tree: "triple",
            self.quadfecta_tree: "quadfecta", # 4복승식 추가
            self.quadcombo_tree: "quadcombo"  # 추가된 quadcombo 처리

        }

        tab_name = tab_names.get(tree, "")
        if not tab_name:
            return

        # 정렬 방향 결정 로직
        if column not in self.high_risk_sort_order[tab_name]:
            self.high_risk_sort_order[tab_name][column] = "desc"
        else:
            self.high_risk_sort_order[tab_name][column] = \
                "asc" if self.high_risk_sort_order[tab_name][column] == "desc" else "desc"

        # 데이터 정렬 및 업데이트 로직
        items = [(tree.set(child, column), child) for child in tree.get_children('')]

        try:
            # 숫자/문자 혼합 정렬 지원
            items.sort(
                key=lambda x: float(x[0].replace('배', '').replace('%', '')) if x[0].replace('.', '', 1).isdigit() else
                x[0].lower(),
                reverse=(self.high_risk_sort_order[tab_name][column] == "desc"))
        except ValueError:
            items.sort(key=lambda x: x[0].lower(),
                       reverse=(self.high_risk_sort_order[tab_name][column] == "desc"))

        # 아이템 재배치
        for index, (val, child) in enumerate(items):
            tree.move(child, '', index)

        # 정렬 방향 표시 업데이트
        self.update_sort_indicator(tree, column, self.high_risk_sort_order[tab_name][column])

    def create_strategy_tabs(self):
        """전략 탭 생성 (고배당 전용 탭 추가)"""
        tab_frame = ttk.Frame(self.root)
        tab_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.strategy_notebook = ttk.Notebook(tab_frame)

        # 전략 테이블 생성
        strategies = [
            ("안정형 조합", 3),
            ("밸런스 조합", 3),
            ("고배당 특화", 3),
            ("변동성 헤징", 4),
            ("롱샷 특화", 2)
        ]

        for strategy, num_horses in strategies:
            frame = ttk.Frame(self.strategy_notebook)
            self.strategy_notebook.add(frame, text=strategy)

            # 컬럼 구성
            columns = ["조합"]
            columns.extend([f"말{i + 1}" for i in range(num_horses)])
            columns.extend(["평균 배당률", "총 베팅 비율", "예상수익률"])

            tree = ttk.Treeview(
                frame,
                columns=columns,
                show="headings",
                height=5
            )

            # 컬럼 설정
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100, anchor="center")

            tree.pack(fill="both", expand=True, padx=5, pady=5)
            self.strategy_trees.append(tree)

        self.strategy_notebook.pack(fill="both", expand=True)

    def create_result_tables(self):
        """결과 테이블 생성"""
        result_frame = ttk.Frame(self.root)
        result_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # 안정적인 추천
        stable_frame = ttk.LabelFrame(result_frame, text="🟢 안정적인 3개 번호 추천")
        stable_frame.pack(side="left", fill="both", expand=True, padx=5)
        self.stable_tree = self.create_result_tree(stable_frame)

        # 고배당 추천
        high_odds_frame = ttk.LabelFrame(result_frame, text="🔵 고배당 4개 번호 추천")
        high_odds_frame.pack(side="left", fill="both", expand=True, padx=5)
        self.high_odds_tree = self.create_result_tree(high_odds_frame)

        # TOP7 추천
        top7_frame = ttk.LabelFrame(result_frame, text="🟡 TOP7 중 4개 랜덤 추천")
        top7_frame.pack(side="left", fill="both", expand=True, padx=5)
        self.top7_tree = self.create_result_tree(top7_frame, show_bet_amount=False)

    def create_result_tree(self, parent, show_bet_amount=True):
        """결과 트리 생성"""
        if show_bet_amount:
            columns = ("순위", "번호", "베팅금액", "비율", "배당률", "안정성", "종합점수")
        else:
            columns = ("순위", "번호", "배당률", "비율", "안정성", "종합점수")

        tree = ttk.Treeview(parent, columns=columns, show="headings", height=5)

        # 컬럼 설정
        col_config = {
            "순위": {"width": 50, "anchor": "center"},
            "번호": {"width": 50, "anchor": "center"},
            "베팅금액": {"width": 100, "anchor": "e"},
            "비율": {"width": 70, "anchor": "center"},
            "배당률": {"width": 80, "anchor": "e"},
            "안정성": {"width": 80, "anchor": "center"},
            "종합점수": {"width": 80, "anchor": "e"}
        }

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, **col_config[col])

        tree.pack(fill="both", expand=True)
        return tree

    def create_buttons(self):
        """버튼 생성"""
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)

        buttons = [
            ("계산 (F5)", self.calculate_odds),
            ("새로 입력 (F2)", self.reset_inputs),
            ("파일 저장 (Ctrl+S)", self.save_to_file),
            ("파일 불러오기 (Ctrl+O)", self.load_from_file),
            ("전략 추천 생성 (F6)", self.generate_strategies),
            ("고위험 배팅 조합 생성 (F7)", self.generate_high_risk_combinations)
        ]

        for i, (text, command) in enumerate(buttons):
            ttk.Button(frame, text=text, command=command).grid(row=0, column=i, padx=5)

        # 단축키 설정
        self.root.bind("<F5>", lambda e: self.calculate_odds())
        self.root.bind("<F6>", lambda e: self.generate_strategies())
        self.root.bind("<F7>", lambda e: self.generate_high_risk_combinations())
        self.root.bind("<F2>", lambda e: self.reset_inputs())
        self.root.bind("<Control-s>", lambda e: self.save_to_file())
        self.root.bind("<Control-o>", lambda e: self.load_from_file())

    def calculate_odds(self):
        """배당률 계산 및 결과 표시 (한국마사회 단승식 공식 적용)"""
        try:
            # 1. 입력값 검증
            total_pool = float(self.total_pool.get())
            if total_pool <= 0:
                raise ValueError("총 베팅 금액은 0보다 커야 합니다.")

            refund_rate = self.refund_rate.get() / 100
            if not 0.7 <= refund_rate <= 0.75:  # 한국마사회 환수율 범위 (70%~75%)
                raise ValueError("환수율은 70%~75% 사이로 입력해주세요")

            # 2. 활성화된 말 필터링 (베팅금액 > 0)
            active_bets = {num: bet for num, bet in self.betting_data.items() if bet > 0}
            if not active_bets:
                raise ValueError("적어도 1개 번호에 베팅해야 합니다.")

            total_active_bet = sum(active_bets.values())
            self.calculated_data = []

            # 3. 한국마사회 단승식 배당률 계산
            for num, bet in active_bets.items():
                # 공식: 배당률 = (총 베팅 금액 × 환수율) / 해당 말의 베팅 금액
                raw_odds = (total_pool * refund_rate) / bet

                # 배당률 소수점 2자리에서 반올림 (마사회 규정)
                odds = round(raw_odds, 2)

                # 배당률 하한선 적용 (최소 1.05배)
                final_odds = max(1.05, odds)

                # 베팅 비율 계산
                bet_ratio = (bet / total_active_bet * 100) if total_active_bet > 0 else 0

                # 안정성 점수 계산 (베팅 금액 비율 기반)
                stability = self.calculate_stability_score(bet, active_bets.values(), refund_rate)

                # 종합 점수 계산 (안정성 + 배당률 가중치)
                composite_score = (stability * 0.6) + (final_odds * 0.4)

                self.calculated_data.append((
                    num, bet, final_odds,
                    self.get_star_representation(stability),
                    bet_ratio, composite_score, stability
                ))

            # 4. UI 업데이트
            self.update_main_table()
            self.update_amount_rank_table(total_active_bet)
            self.update_recommendations()

            # 5. 최종 순위 정보 업데이트
            self.update_ranking_info()

        except Exception as e:
            messagebox.showerror("오류", str(e))

    def calculate_stability_score(self, bet, all_bets, refund_rate):
        """한국마사회 특성 반영한 안정성 점수 계산"""
        max_bet = max(all_bets)
        min_bet = min(all_bets)

        if max_bet == min_bet:
            return 3.0  # 모든 베팅 금액이 동일할 경우

        # 베팅 금액 정규화 (0~1 범위)
        normalized = (bet - min_bet) / (max_bet - min_bet)

        # 환수율 가중치 적용 (높을수록 변동성 증가)
        volatility_factor = 1 + (0.75 - refund_rate) * 2  # 75% 기준 1, 70%일 때 1.1

        # 안정성 점수 계산 (1~5점 범위)
        raw_score = 5 - (normalized * 4 * volatility_factor)
        return max(1.0, min(5.0, round(raw_score, 1)))

    def get_star_representation(self, score):
        """안정성 점수를 별표시로 변환 (한국마사회 스타일)"""
        full_stars = int(score)
        half_star = 1 if (score - full_stars) >= 0.5 else 0
        return "★" * full_stars + "½" * half_star + "☆" * (5 - full_stars - half_star)

    def update_main_table(self):
        """메인 테이블 업데이트"""
        self.tree.delete(*self.tree.get_children())
        for item in self.calculated_data:
            num, bet, odds, stability, _, _, _ = item
            self.tree.insert("", "end", values=(
                num,
                f"{bet:,.2f}",
                f"{odds:.2f}",
                stability
            ))

    def update_amount_rank_table(self, total_active_bet):
        """베팅 금액 순위 테이블 업데이트"""
        active_bets = {num: bet for num, bet in self.betting_data.items() if bet > 0}
        if not active_bets:
            return

        ranked_data = [
            (num, bet, (bet / total_active_bet * 100))
            for num, bet in active_bets.items()
        ]

        reverse_sort = (self.sort_order["amount"] == "desc")
        ranked_data.sort(key=lambda x: x[1], reverse=reverse_sort)

        self.amount_rank_tree.delete(*self.amount_rank_tree.get_children())
        for idx, (num, bet, ratio) in enumerate(ranked_data, 1):
            self.amount_rank_tree.insert("", "end", values=(
                f"{idx}위",
                num,
                f"{bet:,.2f}",
                f"{ratio:.1f}%"
            ))

    def update_recommendations(self):
        """추천 테이블 업데이트"""
        if not self.calculated_data:
            return

        # 안정적인 추천 (종합 점수 순)
        stable_top3 = sorted(
            self.calculated_data,
            key=lambda x: x[5],
            reverse=True
        )[:3]

        # 고배당 추천 (배당률 순)
        high_odds_top4 = sorted(
            self.calculated_data,
            key=lambda x: x[2],
            reverse=True
        )[:4]

        # TOP7 추천
        if len(self.calculated_data) >= 7:
            top7 = sorted(self.calculated_data, key=lambda x: x[5], reverse=True)[:7]
            random_selected = random.sample(top7, min(4, len(top7)))
        else:
            random_selected = []

        self.update_result_table(self.stable_tree, stable_top3)
        self.update_result_table(self.high_odds_tree, high_odds_top4)
        self.update_result_table(self.top7_tree, random_selected)

    def update_result_table(self, tree, data):
        """결과 테이블 업데이트"""
        tree.delete(*tree.get_children())
        for idx, item in enumerate(data, 1):
            num, bet, odds, stability, ratio, composite_score, _ = item

            if tree == self.top7_tree:
                values = (
                    f"{idx}위",
                    num,
                    f"{odds:.2f}",
                    f"{ratio:.1f}%",
                    stability,
                    f"{composite_score:.2f}"
                )
            else:
                values = (
                    f"{idx}위",
                    num,
                    f"{bet:,.2f}",
                    f"{ratio:.1f}%",
                    f"{odds:.2f}",
                    stability,
                    f"{composite_score:.2f}"
                )

            tree.insert("", "end", values=values)

    def generate_strategies(self):
        """전략 추천 생성 - 환수율 구간별 전략 강화"""
        if not self.calculated_data:
            messagebox.showwarning("경고", "먼저 계산을 실행해주세요 (F5)")
            return

        try:
            # 현재 환수율 확인
            refund_rate = self.refund_rate.get()

            # 데이터 정렬 (종합 점수 순)
            sorted_data = sorted(self.calculated_data, key=lambda x: x[5], reverse=True)

            # 배당률 순 정렬
            high_odds_data = sorted(self.calculated_data, key=lambda x: x[2], reverse=True)

            # 1. 안정형 조합 (가중치 적용)
            if len(sorted_data) >= 3:
                stable_top3 = sorted(
                    self.calculated_data,
                    key=lambda x: (x[6] * 0.7) + (x[2] * 0.3),  # 안정성 70%, 배당률 30%
                    reverse=True
                )[:3]
                self._update_strategy_table(
                    self.strategy_trees[0],
                    stable_top3,
                    "안정형"
                )

            # 2. 밸런스 조합 (상위 1개 + 중간 2개)
            if len(sorted_data) >= 3:
                mid_point = len(sorted_data) // 2
                balance_combo = [sorted_data[0]] + sorted_data[mid_point:mid_point + 2]
                self._update_strategy_table(
                    self.strategy_trees[1],
                    balance_combo,
                    "밸런스"
                )

            # 3. 고배당 특화 조합 (환수율 구간별 강화)
            if len(high_odds_data) >= 3:
                if refund_rate <= 75:
                    # 표준: 상위 3개 고배당
                    high_odds_combo = high_odds_data[:3]
                    strategy_name = "표준 고배당"
                elif refund_rate <= 80:
                    # 강화: 상위 4개 고배당
                    high_odds_combo = high_odds_data[:4]
                    strategy_name = "강화 고배당"
                else:
                    # 특화: 상위 2개 초고배당 + 중간 1개
                    high_odds_combo = high_odds_data[:2] + [high_odds_data[len(high_odds_data) // 2]]
                    strategy_name = "초고배당"

                self._update_strategy_table(
                    self.strategy_trees[2],
                    high_odds_combo,
                    strategy_name
                )

            # 4. 변동성 헤징 조합 (고배당 + 안정성 조합)
            if len(sorted_data) >= 4 and len(high_odds_data) >= 2:
                # 중복 없는 조합 생성
                unique_combo = []
                seen_numbers = set()

                # 안정성 부분 선택 (상위 2개 또는 1개)
                if refund_rate <= 75:
                    stable_count = 2
                else:
                    stable_count = 1

                # 안정성 높은 말 선택 (중복 없이)
                for horse in sorted_data:
                    if horse[0] not in seen_numbers:
                        unique_combo.append(horse)
                        seen_numbers.add(horse[0])
                        if len(unique_combo) >= stable_count:
                            break

                # 고배당 말 선택 (중복 없이)
                for horse in high_odds_data:
                    if horse[0] not in seen_numbers:
                        unique_combo.append(horse)
                        seen_numbers.add(horse[0])
                        if len(unique_combo) >= 4:  # 총 4개까지만 선택
                            break

                # 조합이 4개 미만이면 나머지는 종합 점수 높은 말로 채움
                if len(unique_combo) < 4:
                    for horse in sorted_data:
                        if horse[0] not in seen_numbers:
                            unique_combo.append(horse)
                            seen_numbers.add(horse[0])
                            if len(unique_combo) >= 4:
                                break

                self._update_strategy_table(
                    self.strategy_trees[3],
                    unique_combo[:4],  # 최대 4개만 선택
                    "변동성 헤징"
                )

            # 5. 롱샷 특화 조합 (81%+ 전용)
            if refund_rate > 80 and len(high_odds_data) >= 2:
                # 상위 2개 초고배당 말
                longshot_combo = high_odds_data[:2]

                # 5번째 탭에 추가
                if len(self.strategy_trees) > 4:
                    self._update_strategy_table(
                        self.strategy_trees[4],
                        longshot_combo,
                        "롱샷 특화"
                    )

            messagebox.showinfo("완료", f"환수율 {refund_rate}%에 최적화된 전략 추천이 생성되었습니다")

        except Exception as e:
            messagebox.showerror("오류", f"전략 생성 실패:\n{str(e)}")

    def _update_strategy_table(self, tree, horses, strategy_type):
        """전략 테이블 업데이트 (수익률 예측 추가)"""
        tree.delete(*tree.get_children())

        # 평균 배당률 계산
        avg_odds = sum(h[2] for h in horses) / len(horses)

        # 총 베팅 비율 계산
        total_ratio = sum(h[4] for h in horses)

        # 예상 수익률 계산 (환수율 고려)
        refund_rate = self.refund_rate.get() / 100
        expected_return = (avg_odds * refund_rate - 1) * 100  # 백분율

        # 말 번호 목록
        horse_numbers = [h[0] for h in horses]

        # 값 구성
        values = [f"{strategy_type}"]
        values.extend(horse_numbers)
        values.extend([
            f"{avg_odds:.2f}",
            f"{total_ratio:.1f}%",
            f"{expected_return:.1f}%"  # 예상 수익률 추가
        ])

        # 컬럼이 없는 경우 추가
        if len(tree["columns"]) < len(values):
            tree["columns"] = list(tree["columns"]) + ["예상수익률"]
            tree.heading("예상수익률", text="예상수익률")
            tree.column("예상수익률", width=80, anchor="center")

        tree.insert("", "end", values=values)

    def edit_betting(self, event):
        """베팅 금액 편집"""
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        column = self.tree.identify_column(event.x)
        if column != "#2":
            return

        item = self.tree.identify_row(event.y)
        if not item:
            return

        # 기존 편집 엔트리 제거
        if self.current_edit_entry:
            self.current_edit_entry.destroy()
            self.current_edit_entry = None

        current_value = self.tree.item(item, "values")[1]
        x, y, width, height = self.tree.bbox(item, column)

        self.current_edit_entry = ttk.Entry(self.tree)
        self.current_edit_entry.place(x=x, y=y, width=width, height=height, anchor="nw")
        self.current_edit_entry.insert(0, current_value)
        self.current_edit_entry.select_range(0, tk.END)
        self.current_edit_entry.focus()

        self.current_edit_entry.bind("<Return>", lambda e: self.save_edit(item))
        self.current_edit_entry.bind("<FocusOut>", lambda e: self.save_edit(item) if self.current_edit_entry else None)

    def save_edit(self, item):
        """편집 내용 저장"""
        if not self.current_edit_entry:
            return

        try:
            if not self.tree.exists(item):
                messagebox.showwarning("경고", "편집 중인 항목이 더 이상 존재하지 않습니다.")
                return

            new_value = float(self.current_edit_entry.get())
            if new_value < 0:
                raise ValueError("음수는 입력할 수 없습니다.")

            num = int(self.tree.item(item, "values")[0])
            self.betting_data[num] = new_value

            # 트리뷰 업데이트 (안정성 점수는 일단 유지)
            current_values = list(self.tree.item(item, "values"))
            current_values[1] = f"{new_value:,.2f}"
            self.tree.item(item, values=current_values)

        except ValueError as e:
            messagebox.showerror("오류", f"잘못된 입력:\n{str(e)}")
        finally:
            if self.current_edit_entry:
                self.current_edit_entry.destroy()
                self.current_edit_entry = None

    def focus_next_row(self):
        """다음 행으로 이동"""
        current_item = self.tree.focus()
        next_item = self.tree.next(current_item)

        if next_item and self.tree.exists(next_item):
            # 기존 편집 엔트리 제거
            if self.current_edit_entry:
                self.current_edit_entry.destroy()
                self.current_edit_entry = None

            self.tree.focus(next_item)
            self.tree.selection_set(next_item)
            self.edit_selected_row()
        else:
            self.calculate_odds()

    def focus_next_widget(self, event):
        """다음 위젯으로 이동"""
        event.widget.tk_focusNext().focus()
        if self.tree.get_children():
            first_item = self.tree.get_children()[0]
            self.tree.focus(first_item)
            self.tree.selection_set(first_item)
            self.edit_selected_row()
        return "break"

    def sort_amount_rank(self, column):
        """베팅 금액 순위 정렬"""
        if column == "베팅금액":
            self.sort_order["amount"] = "desc" if self.sort_order["amount"] == "asc" else "asc"
            active_bets = {num: bet for num, bet in self.betting_data.items() if bet > 0}
            if active_bets:
                total_active_bet = sum(active_bets.values())
                self.update_amount_rank_table(total_active_bet)

    def reset_inputs(self):
        """입력 초기화 (모든 필드 완전히 초기화)"""
        # 제목은 현재 날짜로 초기화
        self.title_var.set(f"{datetime.now().strftime('%Y%m%d_%H%M')}")

        # 기본값 설정
        self.total_pool.set("10000000")
        self.refund_rate.set(75.0)

        # 베팅 데이터 초기화
        self.betting_data = {i: 0.0 for i in range(1, 16)}
        for var in self.final_rankings:
            var.set("")

        # 모든 테이블 초기화
        for tree in [self.tree, self.stable_tree, self.high_odds_tree,
                     self.top7_tree, self.amount_rank_tree,
                     self.exacta_tree, self.quinella_tree,
                     self.trifecta_tree, self.triple_tree,
                     self.quadfecta_tree, self.quadcombo_tree,
                     self.ranking_tree]:
            tree.delete(*tree.get_children())

        for table in self.strategy_trees:
            table.delete(*table.get_children())

        # 베팅 테이블 초기 데이터
        for num in range(1, 16):
            self.tree.insert("", "end", values=(num, "0.00", 0.0, "★☆☆☆☆"))

        # 상태 초기화
        self.sort_order = {"amount": "desc"}
        self.calculated_data = []
        self.amount_rank_data = []
        self.high_risk_data = []

        # 첫 번째 입력 필드로 포커스 이동
        self.title_entry.focus()

        # 아래 중복된 코드 제거
        # self.sort_order = {"amount": "desc"}
        # self.calculated_data = []
        # self.amount_rank_data = []

    def save_to_file(self):
        """모든 데이터를 파일에 저장 (CSV/JSON 형식)"""
        try:
            # 안전한 파일명 생성
            safe_title = "".join(
                c for c in self.title_var.get()
                if c.isalnum() or c in (' ', '_', '-')
            ).rstrip() or "betting_data"

            default_filename = f"{safe_title}.csv"

            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("JSON Files", "*.json"), ("All Files", "*.*")],
                initialfile=default_filename,
                initialdir=self.last_save_folder
            )

            if not file_path:
                return

            # 저장 폴더 경로 업데이트
            self.last_save_folder = os.path.dirname(file_path)
            self.save_config()

            # 파일 확장자에 따라 저장 방식 결정
            if file_path.lower().endswith('.json'):
                self._save_to_json(file_path)
            else:
                self._save_to_csv(file_path)

            self.current_file_path = file_path
            messagebox.showinfo("저장 완료", f"모든 데이터가 성공적으로 저장되었습니다:\n{file_path}")

        except Exception as e:
            messagebox.showerror("저장 실패", f"파일 저장 중 오류 발생:\n{str(e)}\n\n상세 정보: {traceback.format_exc()}")

    def _save_to_csv(self, file_path):
        """CSV 형식으로 모든 데이터 저장"""
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 1. 메타데이터 섹션
            writer.writerow(["SECTION", "METADATA"])
            writer.writerow(["제목", self.title_var.get()])
            writer.writerow(["총 베팅 금액", self.total_pool.get()])
            writer.writerow(["환수율 (%)", self.refund_rate.get()])

            # 2. 최종 순위 섹션
            writer.writerow([])
            writer.writerow(["SECTION", "FINAL_RANKINGS"])
            for i, var in enumerate(self.final_rankings, 1):
                writer.writerow([f"{i}위", var.get() or ""])

            # 3. 베팅 데이터 섹션
            writer.writerow([])
            writer.writerow(["SECTION", "BETTING_DATA"])
            writer.writerow(["번호", "베팅금액"])
            for num in range(1, 16):
                writer.writerow([num, self.betting_data[num]])

            # 4. 계산된 데이터 섹션
            if self.calculated_data:
                writer.writerow([])
                writer.writerow(["SECTION", "CALCULATED_DATA"])
                writer.writerow(["번호", "베팅금액", "배당률", "안정성", "베팅비율", "종합점수", "안정성점수"])
                for item in self.calculated_data:
                    writer.writerow([
                        item[0], item[1], item[2], item[3],
                        item[4], item[5], item[6]
                    ])

            # 5. 추천 결과 섹션 (안정형, 고배당, TOP7)
            writer.writerow([])
            writer.writerow(["SECTION", "RECOMMENDATIONS"])
            for tree, name in [(self.stable_tree, "안정형"),
                               (self.high_odds_tree, "고배당"),
                               (self.top7_tree, "TOP7")]:
                for child in tree.get_children():
                    writer.writerow([name] + list(tree.item(child)["values"]))

            # 6. 전략 조합 섹션
            writer.writerow([])
            writer.writerow(["SECTION", "STRATEGIES"])
            for i, tree in enumerate(self.strategy_trees):
                strategy_name = self.strategy_notebook.tab(i, "text")
                for child in tree.get_children():
                    writer.writerow([strategy_name] + list(tree.item(child)["values"]))

            # 7. 고위험 배팅 섹션
            writer.writerow([])
            writer.writerow(["SECTION", "HIGH_RISK"])
            for tree, name in [(self.exacta_tree, "복승식"),
                               (self.quinella_tree, "쌍승식"),
                               (self.trifecta_tree, "3복승식"),
                               (self.triple_tree, "트리펙타"),
                               (self.quadfecta_tree, "4복승식"),
                               (self.quadcombo_tree, "4마리조합")]:
                for child in tree.get_children():
                    writer.writerow([name] + list(tree.item(child)["values"]))

    def _save_to_json(self, file_path):
        """JSON 형식으로 모든 데이터 저장"""
        data = {
            "metadata": {
                "title": self.title_var.get(),
                "total_pool": self.total_pool.get(),
                "refund_rate": self.refund_rate.get()
            },
            "final_rankings": [var.get() for var in self.final_rankings],
            "betting_data": self.betting_data,
            "calculated_data": self.calculated_data,

            # 모든 추천 결과 저장
            "recommendations": {
                "stable": [self.stable_tree.item(child)["values"]
                           for child in self.stable_tree.get_children()],
                "high_odds": [self.high_odds_tree.item(child)["values"]
                              for child in self.high_odds_tree.get_children()],
                "top7": [self.top7_tree.item(child)["values"]
                         for child in self.top7_tree.get_children()]
            },

            # 모든 전략 조합 저장
            "strategies": {
                self.strategy_notebook.tab(i, "text"): [
                    tree.item(child)["values"]
                    for child in tree.get_children()
                ]
                for i, tree in enumerate(self.strategy_trees)
            },

            # 모든 고위험 배팅 저장
            "high_risk": {
                "exacta": [self.exacta_tree.item(child)["values"]
                           for child in self.exacta_tree.get_children()],
                "quinella": [self.quinella_tree.item(child)["values"]
                             for child in self.quinella_tree.get_children()],
                "trifecta": [self.trifecta_tree.item(child)["values"]
                             for child in self.trifecta_tree.get_children()],
                "triple": [self.triple_tree.item(child)["values"]
                           for child in self.triple_tree.get_children()],
                "quadfecta": [self.quadfecta_tree.item(child)["values"]
                              for child in self.quadfecta_tree.get_children()],
                "quadcombo": [self.quadcombo_tree.item(child)["values"]
                              for child in self.quadcombo_tree.get_children()]
            }
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load_from_file(self):
        """파일에서 데이터 불러오기 (CSV/JSON 형식)"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("CSV Files", "*.csv"), ("JSON Files", "*.json"), ("All Files", "*.*")],
                initialdir=self.last_save_folder
            )

            if not file_path:
                return

            # 저장 폴더 경로 업데이트
            self.last_save_folder = os.path.dirname(file_path)
            self.save_config()

            # 파일 확장자에 따라 적절한 로드 메서드 호출
            if file_path.lower().endswith('.json'):
                self._load_from_json(file_path)
            else:
                self._load_from_csv(file_path)

            self.current_file_path = file_path
            messagebox.showinfo("불러오기 완료", f"파일에서 데이터를 성공적으로 불러왔습니다:\n{file_path}")

        except Exception as e:
            messagebox.showerror("불러오기 실패", f"파일 불러오기 중 오류 발생:\n{str(e)}\n\n상세 정보: {traceback.format_exc()}")

    def _load_from_csv(self, file_path):
        """CSV 파일에서 모든 데이터 불러오기"""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            sections = {}
            current_section = None

            # 섹션별 데이터 분류
            for row in reader:
                if not row:
                    continue

                if row[0].strip() == "SECTION" and len(row) > 1:
                    current_section = row[1].strip()
                    sections[current_section] = []
                elif current_section is not None:
                    sections[current_section].append(row)

            # 메타데이터 로드
            if "METADATA" in sections:
                for row in sections["METADATA"]:
                    if len(row) >= 2:
                        key, value = row[0], row[1]
                        if key == "제목":
                            self.title_var.set(value)
                        elif key == "총 베팅 금액":
                            self.total_pool.set(value)
                        elif key == "환수율 (%)":
                            self.refund_rate.set(float(value) if value.replace('.', '', 1).isdigit() else 75.0)

            # 최종 순위 로드
            if "FINAL_RANKINGS" in sections:
                for i, row in enumerate(sections["FINAL_RANKINGS"]):
                    if i < len(self.final_rankings) and len(row) >= 2:
                        self.final_rankings[i].set(row[1])

            # 베팅 데이터 로드
            self.betting_data = {i: 0.0 for i in range(1, 16)}
            if "BETTING_DATA" in sections:
                for row in sections["BETTING_DATA"][1:]:  # 헤더 제외
                    try:
                        if len(row) >= 2:
                            num = int(row[0])
                            bet = float(row[1]) if row[1] else 0.0
                            if 1 <= num <= 15:
                                self.betting_data[num] = bet
                    except (ValueError, IndexError):
                        continue

            # 계산된 데이터 로드
            self.calculated_data = []
            if "CALCULATED_DATA" in sections:
                for row in sections["CALCULATED_DATA"][1:]:
                    try:
                        if len(row) >= 7:
                            self.calculated_data.append((
                                int(row[0]),
                                float(row[1]),
                                float(row[2]),
                                row[3],
                                float(row[4]),
                                float(row[5]),
                                float(row[6]) if row[6].replace('.', '', 1).isdigit() else 0
                            ))
                    except (ValueError, IndexError):
                        continue

            # UI 업데이트
            self._update_ui_after_loading()

    def _load_from_json(self, file_path):
        """JSON 파일에서 모든 데이터 불러오기"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            # 메타데이터 로드
            if "metadata" in data:
                meta = data["metadata"]
                self.title_var.set(meta.get("title", ""))
                self.total_pool.set(meta.get("total_pool", "10000000"))
                self.refund_rate.set(float(meta.get("refund_rate", 75.0)))

            # 최종 순위 로드
            if "final_rankings" in data:
                for i, rank in enumerate(data["final_rankings"][:3]):
                    if i < len(self.final_rankings):
                        self.final_rankings[i].set(rank)

            # 베팅 데이터 로드
            self.betting_data = {i: 0.0 for i in range(1, 16)}
            if "betting_data" in data:
                for num, bet in data["betting_data"].items():
                    num = int(num)
                    if 1 <= num <= 15:
                        self.betting_data[num] = float(bet)

            # 계산된 데이터 로드
            self.calculated_data = data.get("calculated_data", [])

            # UI 업데이트
            self._update_ui_after_loading()

            # 추천 결과 로드 (JSON 전용)
            if "recommendations" in data:
                recommendations = data["recommendations"]
                self._load_tree_data(self.stable_tree, recommendations.get("stable", []))
                self._load_tree_data(self.high_odds_tree, recommendations.get("high_odds", []))
                self._load_tree_data(self.top7_tree, recommendations.get("top7", []))

            # 전략 조합 로드 (JSON 전용)
            if "strategies" in data:
                for i, tree in enumerate(self.strategy_trees):
                    tab_name = self.strategy_notebook.tab(i, "text")
                    if tab_name in data["strategies"]:
                        self._load_tree_data(tree, data["strategies"][tab_name])

            # 고위험 배팅 로드 (JSON 전용)
            if "high_risk" in data:
                high_risk = data["high_risk"]
                self._load_tree_data(self.exacta_tree, high_risk.get("exacta", []))
                self._load_tree_data(self.quinella_tree, high_risk.get("quinella", []))
                self._load_tree_data(self.trifecta_tree, high_risk.get("trifecta", []))
                self._load_tree_data(self.triple_tree, high_risk.get("triple", []))
                self._load_tree_data(self.quadfecta_tree, high_risk.get("quadfecta", []))
                self._load_tree_data(self.quadcombo_tree, high_risk.get("quadcombo", []))

    def _update_ui_after_loading(self):
        """데이터 로드 후 UI 업데이트"""
        # 베팅 테이블 업데이트
        self.tree.delete(*self.tree.get_children())
        for num in range(1, 16):
            bet = self.betting_data[num]
            self.tree.insert("", "end", values=(
                num,
                f"{bet:,.2f}" if bet > 0 else "0.00",
                0.0,
                "★☆☆☆☆"
            ))

        # 계산된 데이터가 있으면 UI 업데이트
        if self.calculated_data:
            self.update_main_table()
            total_active_bet = sum(self.betting_data.values())
            if total_active_bet > 0:
                self.update_amount_rank_table(total_active_bet)
            self.update_recommendations()
            self.update_ranking_info()

    def _load_tree_data(self, tree, data):
        """트리뷰에 데이터 로드"""
        tree.delete(*tree.get_children())
        for item in data:
            tree.insert("", "end", values=item)

if __name__ == "__main__":
    root = tk.Tk()
    app = kra_bettingApp(root)
    root.mainloop()
