from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gmm_missing import GMMMissing


NUMERIC_FEATURES = [
    "Aboard",
    "Fatalities",
    "Ground",
    "Year",
    "Month",
    "Day",
]


class ImputationArtifacts:
    """
    Lớp lưu trữ các kết quả và dữ liệu trung gian sau quá trình xử lý và imputation.

    Attributes:
        df_raw (pd.DataFrame): Dữ liệu thô ban đầu từ dataset.
        df_features (pd.DataFrame): Dữ liệu sau khi đã trích xuất các đặc trưng số.
        X_complete (np.ndarray): Ma trận dữ liệu đầy đủ dùng làm mốc so sánh.
        X_missing (np.ndarray): Ma trận dữ liệu đã bị chèn giá trị thiếu (NaN).
        mask (np.ndarray): Mặt nạ Boolean đánh dấu vị trí các giá trị thiếu.
        model (GMMMissing): Mô hình GMM đã được huấn luyện.
        X_imputed (np.ndarray): Dữ liệu sau khi đã được lấp đầy giá trị thiếu.
    """
    def __init__(
        self,
        df_raw: pd.DataFrame,
        df_features: pd.DataFrame,
        X_complete: np.ndarray,
        X_missing: np.ndarray,
        mask: np.ndarray,
        model: GMMMissing,
        X_imputed: np.ndarray,
    ) -> None:
        self.df_raw = df_raw
        self.df_features = df_features
        self.X_complete = X_complete
        self.X_missing = X_missing
        self.mask = mask
        self.model = model
        self.X_imputed = X_imputed


def load_airplane_crashes_dataframe(split: str = "train") -> pd.DataFrame:
    """
    Tải tập dữ liệu tai nạn máy bay từ Hugging Face.

    Args:
        split (str): Phân đoạn dữ liệu cần tải (mặc định là "train").

    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu thô.
    """
    dataset = load_dataset("nateraw/airplane-crashes-and-fatalities", split=split)
    return dataset.to_pandas()


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch và trích xuất các đặc trưng số từ dữ liệu thô.

    Thực hiện chuyển đổi kiểu dữ liệu ngày tháng và ép kiểu số cho các cột liên quan,
    loại bỏ các dòng không có dữ liệu quan sát.

    Args:
        df (pd.DataFrame): DataFrame thô ban đầu.

    Returns:
        pd.DataFrame: Bảng đặc trưng số đã được làm sạch.
    """
    df = df.copy()

    date_parsed = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = date_parsed.dt.year
    df["Month"] = date_parsed.dt.month
    df["Day"] = date_parsed.dt.day

    for col in ["Aboard", "Fatalities", "Ground"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows with at least one observed value to avoid degenerate samples.
    features = df[NUMERIC_FEATURES].dropna(how="all").reset_index(drop=True)
    return features


def inject_mcar_missing(
    X: np.ndarray,
    missing_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chèn giá trị thiếu ngẫu nhiên (MCAR) vào ma trận dữ liệu để đánh giá mô hình.

    Args:
        X (np.ndarray): Ma trận dữ liệu đầy đủ.
        missing_ratio (float): Tỷ lệ giá trị thiếu cần chèn (0.0 đến 1.0).
        random_state (int): Seed để tái lập kết quả ngẫu nhiên.

    Returns:
        tuple[np.ndarray, np.ndarray]: Bao gồm ma trận chứa NaN và mặt nạ vị trí thiếu.
    """
    rng = np.random.default_rng(random_state)
    X_missing = X.copy()
    mask = rng.random(size=X.shape) < missing_ratio

    # Avoid rows that are fully missing.
    all_missing_rows = np.where(mask.all(axis=1))[0]
    for row_id in all_missing_rows:
        keep_col = rng.integers(0, X.shape[1])
        mask[row_id, keep_col] = False

    X_missing[mask] = np.nan
    return X_missing, mask


def fit_gmm_missing(
    X_missing: np.ndarray,
    n_components: int = 4,
    max_iter: int = 150,
    random_state: int = 42,
) -> GMMMissing:
    """
    Khởi tạo và huấn luyện mô hình GMM tùy chỉnh trên dữ liệu có giá trị thiếu.

    Args:
        X_missing (np.ndarray): Ma trận dữ liệu đầu vào chứa NaN.
        n_components (int): Số lượng thành phần Gaussian (clusters).
        max_iter (int): Số vòng lặp tối đa cho thuật toán EM.
        random_state (int): Seed để kiểm soát tính ngẫu nhiên.

    Returns:
        GMMMissing: Đối tượng mô hình đã được huấn luyện.
    """
    model = GMMMissing(
        n_components=n_components,
        max_iter=max_iter,
        tol=1e-5,
        reg_covar=1e-5,
        random_state=random_state,
        init_params="kmeans",
    )
    model.fit(X_missing)
    return model


def evaluate_imputation(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    missing_mask: np.ndarray,
) -> float:
    """
    Tính toán sai số RMSE giữa giá trị thực và giá trị đã được xử lý (imputed).

    Chỉ tính toán dựa trên các vị trí đã bị chèn giá trị thiếu giả lập.

    Args:
        X_true (np.ndarray): Ma trận dữ liệu gốc đầy đủ.
        X_imputed (np.ndarray): Ma trận sau khi đã điền giá trị thiếu.
        missing_mask (np.ndarray): Mặt nạ xác định các vị trí cần so sánh.

    Returns:
        float: Giá trị Root Mean Squared Error.
    """
    y_true = X_true[missing_mask]
    y_pred = X_imputed[missing_mask]
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_pipeline(
    missing_ratio: float = 0.2,
    random_state: int = 42,
    n_components: int = 4,
) -> ImputationArtifacts:
    """
    Thực hiện quy trình khép kín từ tiền xử lý đến điền khuyết dữ liệu bằng GMM.

    Args:
        missing_ratio (float): Tỷ lệ giá trị thiếu giả lập.
        random_state (int): Seed ngẫu nhiên cho toàn bộ pipeline.
        n_components (int): Số cluster cho mô hình GMM.

    Returns:
        ImputationArtifacts: Đối tượng chứa toàn bộ kết quả của quy trình.
    """
    df_raw = load_airplane_crashes_dataframe()
    df_features = build_feature_table(df_raw)

    # Start from complete rows to allow objective RMSE evaluation.
    complete_rows = df_features.dropna().reset_index(drop=True)
    X_complete = complete_rows.to_numpy(dtype=float)

    X_missing, mask = inject_mcar_missing(
        X_complete,
        missing_ratio=missing_ratio,
        random_state=random_state,
    )

    model = fit_gmm_missing(
        X_missing,
        n_components=n_components,
        random_state=random_state,
    )

    return ImputationArtifacts(
        df_raw=df_raw,
        df_features=df_features,
        X_complete=X_complete,
        X_missing=X_missing,
        mask=mask,
        model=model,
        X_imputed=model.X_final_,
    )


def create_visual_report(
    artifacts: ImputationArtifacts,
    output_dir: str | Path = "artifacts",
) -> list[Path]:
    """
    Tạo và lưu trữ các biểu đồ phân tích EDA và kết quả imputation.

    Các biểu đồ bao gồm: tỷ lệ thiếu hụt ban đầu, so sánh phân phối đặc trưng
    và trực quan hóa phân cụm bằng PCA.

    Args:
        artifacts (ImputationArtifacts): Các kết quả từ pipeline.
        output_dir (str | Path): Thư mục đích để lưu file ảnh.

    Returns:
        list[Path]: Danh sách đường dẫn đến các file ảnh đã lưu.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 4))
    missing_rate = artifacts.df_features.isna().mean().sort_values(ascending=False)
    sns.barplot(x=missing_rate.index, y=missing_rate.values, hue=missing_rate.index, ax=ax, palette="Blues_d")
    ax.set_title("Original Missingness Ratio by Feature")
    ax.set_ylabel("Missing Ratio")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p1 = output_path / "missingness_ratio.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    saved_files.append(p1)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for idx, feature in enumerate(NUMERIC_FEATURES):
        ax = axes[idx]
        sns.kdeplot(artifacts.X_complete[:, idx], label="Ground Truth", ax=ax, linewidth=2)
        sns.kdeplot(artifacts.X_imputed[:, idx], label="GMM Imputed", ax=ax, linewidth=2)
        ax.set_title(feature)
        if idx == 0:
            ax.legend()
    fig.suptitle("Feature Distribution: Ground Truth vs GMM-Imputed", fontsize=13)
    fig.tight_layout()
    p2 = output_path / "distribution_comparison.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    saved_files.append(p2)

    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(artifacts.X_imputed)
    labels = artifacts.model.predict()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        x=proj[:, 0],
        y=proj[:, 1],
        hue=labels,
        palette="tab10",
        s=30,
        ax=ax,
        legend="full",
    )
    ax.set_title("GMM Clusters on PCA Projection (Imputed Data)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    p3 = output_path / "pca_clusters.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    saved_files.append(p3)

    return saved_files


def main() -> None:
    """
    Hàm thực thi chính: chạy pipeline, đánh giá và xuất báo cáo hình ảnh.
    """
    artifacts = run_pipeline(missing_ratio=0.2, random_state=42, n_components=4)
    rmse = evaluate_imputation(artifacts.X_complete, artifacts.X_imputed, artifacts.mask)
    figures = create_visual_report(artifacts)

    print("Airplane Crash Dataset - GMM Missing-Data Imputation")
    print(f"Samples used (complete rows): {artifacts.X_complete.shape[0]}")
    print(f"Features: {NUMERIC_FEATURES}")
    print(f"RMSE on simulated missing cells: {rmse:.4f}")
    print("Saved figures:")
    for fig in figures:
        print(f" - {fig}")


if __name__ == "__main__":
    main()