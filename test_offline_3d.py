import open3d as o3d
import numpy as np
import os
from typing import List, Dict, Any, Optional

# --- (THAY ĐỔI) IMPORT THƯ VIỆN MỚI ---
import cv2
from ultralytics import YOLO
# -------------------------------------

# (MỚI) Thêm thư viện để kiểm tra điểm trong đa giác (point-in-polygon)
from matplotlib.path import Path 

import csv
import sys
try:
    import keyboard  # Thư viện để bắt phím mũi tên
except ImportError:
    print("LỖI: Không tìm thấy thư viện 'keyboard'.")
    print("Vui lòng chạy: pip install keyboard")
    sys.exit(1)

# =============================================================================
# --- 1. LỚP CẤU HÌNH (CONFIG) ---
# =============================================================================

class PipelineConfig:
    """
    Lưu trữ tất cả các tham số cố định cho pipeline xử lý.
    """
    def __init__(self):
        
        # --- 1.1. Cấu hình Đường dẫn ---
        self.BASE_DIR = "./submit"
        self.RGB_DIR = os.path.join(self.BASE_DIR, "rgb")
        self.DEPTH_DIR = os.path.join(self.BASE_DIR, "depth")

        # --- 1.2. Tham số Camera (Nội tại) ---
        self.WIDTH = 1280
        self.HEIGHT = 720
        self.FX_COLOR = 643.90087890625
        self.FY_COLOR = 643.1365356445312
        self.CX_COLOR = 650.2113037109375
        self.CY_COLOR = 355.79559326171875
        self.INTRINSIC_COLOR = o3d.camera.PinholeCameraIntrinsic(
            self.WIDTH, self.HEIGHT, self.FX_COLOR, self.FY_COLOR, self.CX_COLOR, self.CY_COLOR
        )

        # --- 1.3. Tham số Xử lý PCD ---
        self.DEPTH_SCALE = 1000.0
        self.DEPTH_TRUNC = 5.0

        # --- 1.4. Tham số Lọc (Hộp) ---
        MIN_BOUND_CROP = np.array([-0.4, -0.45, 0.7])
        MAX_BOUND_CROP = np.array([ 0.35,  0.25,  1.175])
        self.CROPPING_BOX = o3d.geometry.AxisAlignedBoundingBox(MIN_BOUND_CROP, MAX_BOUND_CROP)

        PLANE_MIN_BOUND = np.array([-0.4, -0.37, 1.1])
        PLANE_MAX_BOUND = np.array([ -0.15,  0.25,  1.175])
        self.REMOVAL_BOX_1 = o3d.geometry.AxisAlignedBoundingBox(PLANE_MIN_BOUND, PLANE_MAX_BOUND)

        PLANE_MIN_BOUND_2 = np.array([-0.4, -0.45, 1.175])
        PLANE_MAX_BOUND_2 = np.array([ -0.15,  -0.35,  1.175])
        self.REMOVAL_BOX_2 = o3d.geometry.AxisAlignedBoundingBox(PLANE_MIN_BOUND_2, MAX_BOUND_CROP)

        # --- 1.5. Tham số Lọc (Thuật toán) ---
        self.SOR_NB_NEIGHBORS = 20
        self.SOR_STD_RATIO = 2.0
        self.NORMAL_EST_RADIUS = 0.02
        self.NORMAL_EST_MAX_NN = 30
        self.NORMAL_UP_VECTOR = np.array([0, 0, -1])
        self.NORMAL_ANGLE_THRESHOLD_DEG = 60.0
        self.NORMAL_DOT_PRODUCT_THRESHOLD = np.cos(np.deg2rad(self.NORMAL_ANGLE_THRESHOLD_DEG))

        # --- 1.6. Tham số Phân cụm (DBSCAN) ---
        self.DBSCAN_EPS = 0.02
        self.DBSCAN_MIN_POINTS = 20
        self.MIN_CLUSTER_EXTENT = 0.05 

        # --- 1.7. Tham số RANSAC ---
        self.RANSAC_DISTANCE_THRESHOLD = 0.05
        self.RANSAC_MIN_POINTS = 1000

        # --- 1.8. Tham số Chọn lọc Ứng viên ---
        self.SELECTION_MIN_X = -0.16
        self.SELECTION_Z_TIE_THRESHOLD = 0.005
        
        # --- 1.9. Tham số ROI (Hộp xanh lá) ---
        self.ROI_U_MIN = 560
        self.ROI_V_MIN = 150
        self.ROI_U_MAX = 560 + 300
        self.ROI_V_MAX = 150 + 330
        self.ROI_Z_REFERENCE = 1.0 # Z tham chiếu để chiếu 2D -> 3D
        self.ROI_Z_POSITION = 0.8
        self.ROI_BOX_HEIGHT_Z = 0.4
        
        # --- 1.10. Tham số Model Segmentation 2D (YOLO) ---
        # (THAY ĐỔI) Sử dụng đường dẫn local thay vì API Key
        self.YOLO_MODEL_PATH = "./best5.pt" 
        self.SEG_CONFIDENCE = 0.40 
        
        # --- 1.11. Tham số Chiếu 3D (Prism) ---
        self.SEG_PRISM_Z_MIN = 0.7 
        self.SEG_PRISM_Z_MAX = 1.175


# =============================================================================
# --- 2. HÀM TIỆN ÍCH (UTILITIES) ---
# =============================================================================

def create_sphere(center, color, radius=0.02):
    """Tạo một đối tượng mesh hình cầu tại tâm (center) với màu (color)."""
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere

def unproject_point_to_xy(u, v, Z, fx, fy, cx, cy):
    """Chuyển đổi tọa độ ảnh (u, v, Z) sang tọa độ 3D (X, Y)."""
    X = (u - cx) * Z / fx
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return (X, Y)


# =============================================================================
# --- 3. LỚP XỬ LÝ CHÍNH (PROCESSOR CLASS) ---
# =============================================================================

class PointCloudProcessor:
    """
    Quản lý toàn bộ quy trình: tải, xử lý, tìm kiếm và trực quan hóa
    đám mây điểm cho MỘT tệp ảnh duy nhất.
    """

    def __init__(self, config: PipelineConfig, file_base_name: str, target_coords: List[float], model: Any):
        self.config = config
        self.file_base_name = file_base_name
        self.file_name = f"{file_base_name}.png"
        self.target_coords_original = np.array(target_coords) 

        self.model = model # Đây sẽ là YOLO model object
        self.color_file = os.path.join(self.config.RGB_DIR, self.file_name)
        self.depth_file = os.path.join(self.config.DEPTH_DIR, self.file_name)
        
        # Trạng thái 3D
        self.pcd_original: Optional[o3d.geometry.PointCloud] = None
        self.pcd_processed: Optional[o3d.geometry.PointCloud] = None
        self.candidates: List[Dict[str, Any]] = []
        self.final_choice: Optional[Dict[str, Any]] = None
        
        # Đối tượng để vẽ
        self.geometries_to_draw: List[o3d.geometry.Geometry] = []
        self.num_valid_clusters = 0

        # Trạng thái phân vùng 2D/3D
        self.prism_indices_list: List[np.ndarray] = []
        self.outside_indices: Optional[np.ndarray] = None
        self.prisms_for_drawing: List[o3d.geometry.LineSet] = []


    def run_pipeline(self):
        """Thực thi toàn bộ pipeline xử lý."""
        print(f"\n--- Đang xử lý tệp: {self.file_name} ---")
        print(f"Tọa độ điểm gốc (CSV): {self.target_coords_original}") 
        
        try:
            if not self._load_data():
                return
            
            # B1: Lọc 3D trước
            self._preprocess_pcd()
            
            if not self.pcd_processed or not self.pcd_processed.has_points():
                print("Lỗi: Không còn điểm nào sau khi tiền xử lý. Bỏ qua.")
                return

            # B2: Chạy model YOLO 2D để PHÂN VÙNG pcd_processed
            self._run_2d_segmentation_and_project()

            # B3: Chạy DBSCAN/RANSAC trên TỪNG VÙNG đã phân
            self._find_plane_candidates()
            
            # B4: Chọn ứng viên tốt nhất từ tất cả các vùng
            self._select_best_candidate()
            
            if self.final_choice:
                self._compare_with_target()

        except Exception as e:
            print(f"[LỖI NGHIÊM TRỌNG] Xảy ra lỗi khi chạy pipeline: {e}")
            import traceback
            traceback.print_exc()

    def _load_data(self) -> bool:
        """Tải ảnh RGB và Depth, tạo PointCloud ban đầu."""
        if not os.path.exists(self.color_file) or not os.path.exists(self.depth_file):
            print(f"Lỗi: Không tìm thấy {self.color_file} hoặc {self.depth_file}. Bỏ qua.")
            return False

        color_raw = o3d.io.read_image(self.color_file)
        depth_raw = o3d.io.read_image(self.depth_file)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw,
            depth_scale=self.config.DEPTH_SCALE,
            depth_trunc=self.config.DEPTH_TRUNC,
            convert_rgb_to_intensity=False
        )
        
        self.pcd_original = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.config.INTRINSIC_COLOR
        )
        print(f"Tải dữ liệu thành công, có {len(self.pcd_original.points)} điểm ban đầu.")
        return True

    # =========================================================================
    # --- (THAY ĐỔI LỚN) CÁC HÀM XỬ LÝ MODEL YOLO 2D ---
    # =========================================================================

    def _run_2d_segmentation_and_project(self):
        """
        Chạy model YOLOv8. Dùng kết quả (Masks) để TÁCH pcd_processed thành các
        nhóm indices: "trong prism 1", "trong prism 2", ..., "bên ngoài".
        """
        if self.model is None:
            print("[Cảnh báo] Không có model YOLO. Toàn bộ điểm sẽ được xử lý chung.")
            all_indices = np.arange(len(self.pcd_processed.points))
            self.outside_indices = all_indices
            return
            
        if not self.pcd_processed or not self.pcd_processed.has_points():
            print("Lỗi: Không có điểm 3D đã xử lý để chạy phân vùng 2D.")
            return

        print(f"Đang chạy YOLOv8 Segmentation...")
        
        try:
            # --- THAY ĐỔI: Chạy Inference bằng YOLOv8 ---
            results = self.model(
                self.color_file, 
                conf=self.config.SEG_CONFIDENCE,
                verbose=False
            )
            result = results[0] # Lấy kết quả ảnh đầu tiên
            
            # Kiểm tra xem có mask nào không
            if result.masks is None:
                print(" -> YOLO: Không phát hiện đối tượng nào. Toàn bộ điểm sẽ được xử lý chung.")
                all_indices = np.arange(len(self.pcd_processed.points))
                self.outside_indices = all_indices
                return

            # Lấy các đường bao đa giác (polygon coordinates)
            # result.masks.xy trả về một list các array numpy, mỗi array là các điểm (x, y) của contour
            contours = result.masks.xy 
            print(f" -> YOLO: Phát hiện {len(contours)} đối tượng (Masks).")

            # --- PHẦN LOGIC 3D (GIỮ NGUYÊN, CHỈ SỬA ĐẦU VÀO) ---
            
            # Lấy tất cả các điểm 3D đã được tiền xử lý
            all_points_3d = np.asarray(self.pcd_processed.points)
            all_indices = np.arange(len(all_points_3d))
            used_indices_mask = np.zeros(len(all_points_3d), dtype=bool)

            # Lấy tham số camera
            fx, fy = self.config.FX_COLOR, self.config.FY_COLOR
            cx, cy = self.config.CX_COLOR, self.config.CY_COLOR
            
            # Chiếu tất cả các điểm 3D về 2D (u, v) CÙNG MỘT LÚC
            X = all_points_3d[:, 0]
            Y = all_points_3d[:, 1]
            Z = all_points_3d[:, 2]
            
            Z_safe = np.where(Z == 0, 1e-6, Z)
            
            u = (X * fx / Z_safe) + cx
            v = (Y * fy / Z_safe) + cy
            projected_points_2d = np.vstack((u, v)).T

            # Lặp qua các contour (mask) từ YOLO
            for i, contour in enumerate(contours):
                # contour là numpy array shape (N, 2)
                
                # --- CHUYỂN ĐỔI: YOLO numpy -> List of Dicts (cho hàm vẽ _create_mask_prism) ---
                mask_points_2d_list = [{'x': p[0], 'y': p[1]} for p in contour]
                
                # 1. (VẼ) Tạo đường bao 3D (prism) từ mask 2D để VẼ
                prism_lineset = self._create_mask_prism(mask_points_2d_list)
                self.prisms_for_drawing.append(prism_lineset)
                
                # 2. (LOGIC) Tạo Path 2D từ mask (dùng trực tiếp numpy array từ YOLO)
                if len(contour) < 3:
                    continue 
                
                path = Path(contour) # Matplotlib Path nhận numpy array (N,2)

                # 3. (LOGIC) Tìm các điểm 3D nằm trong đa giác 2D VÀ trong khoảng Z
                
                # Kiểm tra 2D: Điểm 2D nào nằm trong đa giác?
                is_inside_2d_polygon = path.contains_points(projected_points_2d)
                
                # Kiểm tra 3D: Điểm 3D nào nằm trong khoảng Z?
                is_within_z_range = (Z >= self.config.SEG_PRISM_Z_MIN) & \
                                    (Z <= self.config.SEG_PRISM_Z_MAX)
                
                # Kết hợp: Điểm nào thỏa mãn cả 2 VÀ chưa được gán?
                final_mask_for_this_prism = is_inside_2d_polygon & \
                                            is_within_z_range & \
                                            (~used_indices_mask)
                
                # Lấy indices của các điểm này
                indices_for_this_prism = all_indices[final_mask_for_this_prism]
                
                if len(indices_for_this_prism) > 0:
                    self.prism_indices_list.append(indices_for_this_prism)
                    # Đánh dấu các điểm này là đã sử dụng
                    used_indices_mask[indices_for_this_prism] = True
                    print(f"   -> Phân vùng 'Prism {i+1}': {len(indices_for_this_prism)} điểm.")
                else:
                    print(f"   -> Phân vùng 'Prism {i+1}': 0 điểm (có thể đã bị lọc hoặc nằm ngoài khoảng Z).")

            # Sau khi lặp xong, các điểm KHÔNG được gán sẽ là "bên ngoài"
            self.outside_indices = all_indices[~used_indices_mask]
            print(f"   -> Phân vùng 'Bên ngoài': {len(self.outside_indices)} điểm.")

        except Exception as e:
            print(f"[LỖI] Đã xảy ra lỗi trong quá trình phân vùng 2D/3D: {e}")
            import traceback
            traceback.print_exc()
            # Nếu lỗi, coi như tất cả là "bên ngoài"
            all_indices = np.arange(len(self.pcd_processed.points))
            self.outside_indices = all_indices

    def _create_mask_prism(self, mask_points_2d: List[Dict[str, float]]) -> o3d.geometry.LineSet:
        """
        Tạo một 'hình lăng trụ' (prism) 3D từ một mask 2D.
        (Hàm này chỉ dùng để VẼ, không ảnh hưởng logic)
        """
        Z_ref = self.config.ROI_Z_REFERENCE 
        Z_front = self.config.SEG_PRISM_Z_MIN
        Z_back = self.config.SEG_PRISM_Z_MAX
        fx, fy = self.config.FX_COLOR, self.config.FY_COLOR
        cx, cy = self.config.CX_COLOR, self.config.CY_COLOR
        
        xy_coords_at_ref = []
        for p in mask_points_2d:
            u, v = p['x'], p['y']
            X, Y = unproject_point_to_xy(u, v, Z_ref, fx, fy, cx, cy)
            xy_coords_at_ref.append((X, Y))
            
        points_3d = []
        for X, Y in xy_coords_at_ref: points_3d.append([X, Y, Z_front])
        for X, Y in xy_coords_at_ref: points_3d.append([X, Y, Z_back])
            
        n = len(xy_coords_at_ref)
        if n == 0: return o3d.geometry.LineSet()

        lines = []
        for i in range(n):
            i_next = (i + 1) % n
            lines.append([i, i_next])
            lines.append([i + n, i_next + n])
            lines.append([i, i + n])

        prism_lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_3d),
            lines=o3d.utility.Vector2iVector(lines),
        )
        prism_lineset.paint_uniform_color([1.0, 0.0, 0.0]) # Màu đỏ
        return prism_lineset

    # =========================================================================
    # --- CÁC HÀM XỬ LÝ 3D (NHƯ CŨ) ---
    # =========================================================================

    def _preprocess_pcd(self):
        """Áp dụng các bước lọc, cắt và loại bỏ nhiễu."""
        if not self.pcd_original:
            return

        print("Đang tiền xử lý 3D (Cắt, Xóa, Lọc nhiễu, Lọc pháp tuyến)...")
        
        pcd_cropped = self.pcd_original.crop(self.config.CROPPING_BOX)
        
        indices_remove_1 = self.config.REMOVAL_BOX_1.get_point_indices_within_bounding_box(pcd_cropped.points)
        pcd_interm = pcd_cropped.select_by_index(indices_remove_1, invert=True)

        indices_remove_2 = self.config.REMOVAL_BOX_2.get_point_indices_within_bounding_box(pcd_interm.points)
        pcd_filtered = pcd_interm.select_by_index(indices_remove_2, invert=True)
        
        print(f"  Số điểm trước khi lọc nhiễu: {len(pcd_filtered.points)}")
        if len(pcd_filtered.points) == 0:
             self.pcd_processed = pcd_filtered
             return

        cl, ind = pcd_filtered.remove_statistical_outlier(
            nb_neighbors=self.config.SOR_NB_NEIGHBORS,
            std_ratio=self.config.SOR_STD_RATIO
        )
        pcd_filtered = pcd_filtered.select_by_index(ind) 
        print(f"  Số điểm sau khi lọc nhiễu: {len(pcd_filtered.points)}")
        
        if not pcd_filtered.has_points():
            self.pcd_processed = pcd_filtered
            return
            
        pcd_filtered.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.NORMAL_EST_RADIUS, 
                max_nn=self.config.NORMAL_EST_MAX_NN
            )
        )
        pcd_filtered.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
        
        normals = np.asarray(pcd_filtered.normals)
        dot_products = np.dot(normals, self.config.NORMAL_UP_VECTOR)
        indices_to_keep = np.where(dot_products > self.config.NORMAL_DOT_PRODUCT_THRESHOLD)[0]
        
        print(f"  Số điểm trước khi lọc 'Tấm Màn': {len(pcd_filtered.points)}")
        pcd_filtered = pcd_filtered.select_by_index(indices_to_keep)
        print(f"  Số điểm sau khi lọc 'Tấm Màn': {len(pcd_filtered.points)}")
        
        self.pcd_processed = pcd_filtered

    def _find_plane_candidates(self):
        """Chạy DBSCAN và RANSAC RIÊNG LẺ trên từng nhóm indices."""
        if not self.pcd_processed:
            print("Lỗi: Không có điểm đã xử lý để phân cụm.")
            return

        self.candidates = []
        self.num_valid_clusters = 0
        geometries_clusters = [] 
        
        all_index_groups = []
        group_names = []
        
        for i, prism_indices in enumerate(self.prism_indices_list):
            if len(prism_indices) > self.config.DBSCAN_MIN_POINTS:
                all_index_groups.append(prism_indices)
                group_names.append(f"Prism {i+1}")
            
        if self.outside_indices is not None and len(self.outside_indices) > self.config.DBSCAN_MIN_POINTS:
            all_index_groups.append(self.outside_indices)
            group_names.append("Bên ngoài (Outside)")

        if not all_index_groups:
            print("Không có nhóm điểm nào đủ lớn để chạy DBSCAN.")
            return

        print("\n" + "="*30)
        print("PHẦN 4: CHẠY PHÂN CỤM RIÊNG LẺ TRÊN TỪNG VÙNG")

        for i, index_group in enumerate(all_index_groups):
            group_name = group_names[i]
            print(f"--- Đang xử lý vùng '{group_name}' ({len(index_group)} điểm) ---")
            
            pcd_subset = self.pcd_processed.select_by_index(index_group)
            if not pcd_subset.has_points():
                continue

            print(f"  Đang chạy DBSCAN (eps={self.config.DBSCAN_EPS}, min_points={self.config.DBSCAN_MIN_POINTS})...")
            labels = np.array(pcd_subset.cluster_dbscan(
                eps=self.config.DBSCAN_EPS, 
                min_points=self.config.DBSCAN_MIN_POINTS, 
                print_progress=False
            ))
            
            unique_labels = np.unique(labels)
            num_clusters_in_group = len(unique_labels[unique_labels != -1])
            
            if num_clusters_in_group == 0:
                 print(f"  -> Vùng '{group_name}': Không tìm thấy cụm nào (chỉ có nhiễu).")
                 pcd_noise = pcd_subset
                 pcd_noise.paint_uniform_color([0.5, 0.5, 0.5])
                 geometries_clusters.append(pcd_noise)
                 continue

            print(f"  -> Vùng '{group_name}': Tìm thấy {num_clusters_in_group} cụm (bỏ qua nhiễu -1).")
            
            for label in unique_labels:
                indices_in_subset = np.where(labels == label)[0]
                cluster_pcd = pcd_subset.select_by_index(indices_in_subset)

                if label == -1:
                    cluster_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Xám (nhiễu)
                    geometries_clusters.append(cluster_pcd)
                    continue

                candidate_info, cluster_geoms = self._process_cluster(cluster_pcd, label)
                
                geometries_clusters.extend(cluster_geoms)
                if candidate_info:
                    self.candidates.append(candidate_info)
                    self.num_valid_clusters += 1
            
            print(f"--- Kết thúc xử lý vùng '{group_name}' ---")


        print(f"\n*** Tóm tắt phân cụm: Tìm thấy TỔNG CỘNG {self.num_valid_clusters} cụm hợp lệ từ {len(all_index_groups)} vùng ***")
        self.geometries_to_draw.extend(geometries_clusters)

    def _process_cluster(self, cluster_pcd: o3d.geometry.PointCloud, label: int):
        """Xử lý cụm: RANSAC và tìm tâm thực tế (Inlier)."""
        geometries = []
        
        aabb = cluster_pcd.get_axis_aligned_bounding_box()
        if np.max(aabb.get_extent()) <= self.config.MIN_CLUSTER_EXTENT:
            print(f"   - Cụm {label}: BỎ QUA ({len(cluster_pcd.points)} điểm). Kích thước quá nhỏ.")
            cluster_pcd.paint_uniform_color([0.5, 0.5, 0.5]) 
            geometries.append(cluster_pcd)
            return None, geometries

        print(f"   + Cụm {label}: HỢP LỆ ({len(cluster_pcd.points)} điểm). Đang chạy RANSAC...")

        try:
            plane_model, inlier_indices = cluster_pcd.segment_plane(
                distance_threshold=self.config.RANSAC_DISTANCE_THRESHOLD,
                ransac_n=3,
                num_iterations=1000
            )
        
            if len(inlier_indices) < self.config.RANSAC_MIN_POINTS:
                print(f"      -> RANSAC: Mặt phẳng quá nhỏ ({len(inlier_indices)} điểm). Bỏ qua.")
                cluster_pcd.paint_uniform_color([1.0, 0.5, 0.0])
                geometries.append(cluster_pcd)
                return None, geometries
            
            print(f"      -> RANSAC: Tìm thấy mặt phẳng ({len(inlier_indices)} điểm).")
            
            plane_pcd = cluster_pcd.select_by_index(inlier_indices)
            outlier_pcd = cluster_pcd.select_by_index(inlier_indices, invert=True)
            
            plane_pcd.paint_uniform_color([1.0, 0.0, 1.0]) # Tím (Mặt phẳng)
            outlier_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Xám (Nhiễu của cụm)
            geometries.append(plane_pcd)
            geometries.append(outlier_pcd)

            # --- LOGIC TÌM TÂM INLIER ---
            inlier_points = np.asarray(plane_pcd.points)
            centroid = np.mean(inlier_points, axis=0) 
            
            distances = np.linalg.norm(inlier_points - centroid, axis=1)
            closest_inlier_index = np.argmin(distances)
            actual_inlier_center = inlier_points[closest_inlier_index]
            
            # Lấy vector pháp tuyến
            original_normal = np.array(plane_model[:3])
            norm = np.linalg.norm(original_normal)
            if norm < 1e-6:
                normal_unit = np.array([0, 0, -1]) 
            else:
                normal_unit = original_normal / norm

            # Vẽ quả cầu tại TÂM THỰC TẾ
            center_sphere = create_sphere(actual_inlier_center, [0.0, 1.0, 1.0], radius=0.025) # Xanh lơ
            geometries.append(center_sphere)
            
            print(f"      -> Centroid (lý tưởng):    {np.round(centroid, 5)}")
            print(f"      -> TÂM INLIER (thực tế): {np.round(actual_inlier_center, 5)}")

            candidate_info = {
                "center_original": actual_inlier_center, 
                "normal_unit_original": normal_unit
            }
            return candidate_info, geometries
            
        except Exception as e:
            print(f"      -> Lỗi RANSAC: {e}. Bỏ qua cụm.")
            cluster_pcd.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(cluster_pcd)
            return None, geometries

    def _select_best_candidate(self):
        """Chọn ứng viên tốt nhất theo logic."""
        print("\n" + "="*30)
        print("PHẦN 5: CHỌN LỌC TÂM CUỐI CÙNG (TRÊN TỌA ĐỘ GỐC)")
        print(f"Tìm thấy tổng cộng {len(self.candidates)} ứng viên từ tất cả các vùng.")

        if not self.candidates:
            print("[LỖI] Không có ứng viên nào được tìm thấy. Dừng chọn lọc.")
            print("="*30 + "\n")
            return

        filtered_candidates = []
        for c in self.candidates:
            x_coord = c["center_original"][0]
            if x_coord >= self.config.SELECTION_MIN_X:
                filtered_candidates.append(c)
            else:
                print(f"   [LỌAI] Ứng viên {np.round(c['center_original'], 3)} vì X ({x_coord:.3f}) < {self.config.SELECTION_MIN_X}")
        
        print(f"Còn lại {len(filtered_candidates)} ứng viên sau khi lọc X >= {self.config.SELECTION_MIN_X}.")

        if not filtered_candidates:
            print("[LỖI] Không tìm thấy ứng viên nào thỏa mãn điều kiện X.")
            print("="*30 + "\n")
            return

        sorted_candidates = sorted(
            filtered_candidates, 
            key=lambda c: abs(c["center_original"][2]) 
        )
        
        best_candidate_by_z = sorted_candidates[0]
        best_z_abs = abs(best_candidate_by_z["center_original"][2])
        print(f"   [INFO] |Z| nhỏ nhất tìm thấy: {best_z_abs:.4f} (từ tâm {np.round(best_candidate_by_z['center_original'], 3)})")

        tie_breaker_group = [
            c for c in sorted_candidates 
            if abs(abs(c["center_original"][2]) - best_z_abs) <= self.config.SELECTION_Z_TIE_THRESHOLD
        ]
        
        print(f"   [INFO] Tìm thấy {len(tie_breaker_group)} ứng viên trong nhóm Z-tie-break (chênh lệch <= {self.config.SELECTION_Z_TIE_THRESHOLD}).")

        if len(tie_breaker_group) > 1:
            print("   [INFO] Có > 1 ứng viên. Áp dụng tie-break: Chọn X nhỏ nhất.")
            final_choice = sorted(
                tie_breaker_group, 
                key=lambda c: c["center_original"][0], 
            )[0]
        else:
            print("   [INFO] Chỉ có 1 ứng viên, không cần tie-break.")
            final_choice = best_candidate_by_z

        print("\n" + "!"*30)
        print("!!! LỰA CHỌN CUỐI CÙNG !!!")
        final_coords = final_choice["center_original"]
        final_normal = final_choice["normal_unit_original"]
        print(f"   Tọa độ (GỐC): {np.round(final_coords, 5)}")
        print(f"   Vecto pháp tuyến (GỐC): {np.round(final_normal, 5)}")
        print("!"*30 + "\n")

        self.final_choice = final_choice
        print("="*30 + "\n")

    def _compare_with_target(self):
        """So sánh kết quả cuối cùng với tọa độ CSV."""
        if not self.final_choice:
            return

        print("--- SO SÁNH VỚI TÂM TỪ CSV (TRÊN TỌA ĐỘ GỐC) ---")
        
        csv_coords_original = self.target_coords_original
        final_center_coords = self.final_choice["center_original"]
        
        print(f"   Tâm CSV (dự phòng, gốc): {np.round(csv_coords_original, 5)}")
        print(f"   Tâm tìm thấy (gốc):        {np.round(final_center_coords, 5)}")
        
        distance = np.linalg.norm(final_center_coords - csv_coords_original)
        print(f"   -> Khoảng cách (L2): {distance:.5f} mét")
        print("--- KẾT THÚC SO SÁNH ---")

    def _create_visualization_geometries(self):
        """Tạo các hộp/đường viền (boxes, linesets) CẤU HÌNH."""
        
        # C1. Hộp ROI (Xanh lá)
        roi_corners_2d = [
            (self.config.ROI_U_MIN, self.config.ROI_V_MIN),
            (self.config.ROI_U_MAX, self.config.ROI_V_MIN),
            (self.config.ROI_U_MAX, self.config.ROI_V_MAX),
            (self.config.ROI_U_MIN, self.config.ROI_V_MAX)
        ]
        
        reference_xy = []
        for u, v in roi_corners_2d:
            X, Y = unproject_point_to_xy(
                u, v, self.config.ROI_Z_REFERENCE, 
                self.config.FX_COLOR, self.config.FY_COLOR, 
                self.config.CX_COLOR, self.config.CY_COLOR
            )
            reference_xy.append((X, Y))

        points_3d = []
        Z_front = self.config.ROI_Z_POSITION
        Z_back = Z_front + self.config.ROI_BOX_HEIGHT_Z
        for X, Y in reference_xy: points_3d.append([X, Y, Z_front])
        for X, Y in reference_xy: points_3d.append([X, Y, Z_back])

        lines = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]
        
        roi_box = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_3d),
            lines=o3d.utility.Vector2iVector(lines),
        )
        roi_box.paint_uniform_color([0.0, 1.0, 0.0]) # Xanh lá
        self.geometries_to_draw.append(roi_box)

        # C2. Hộp Cắt (Xanh dương)
        crop_box_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(self.config.CROPPING_BOX)
        crop_box_lines.paint_uniform_color([0.0, 0.0, 1.0]) 
        self.geometries_to_draw.append(crop_box_lines)

        # C3. Hộp Xóa 1 (Đen)
        remove_box_1_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(self.config.REMOVAL_BOX_1)
        remove_box_1_lines.paint_uniform_color([0.0, 0.0, 0.0]) 
        self.geometries_to_draw.append(remove_box_1_lines)
        
        # C4. Hộp Xóa 2 (Tím/Magenta)
        remove_box_2_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(self.config.REMOVAL_BOX_2)
        remove_box_2_lines.paint_uniform_color([1.0, 0.0, 1.0]) 
        self.geometries_to_draw.append(remove_box_2_lines)

        # C5. Quả cầu VÀNG LỚN (Lựa chọn cuối cùng)
        if self.final_choice:
            print("Đang vẽ quả cầu VÀNG LỚN (lựa chọn cuối cùng)...")
            center_original = self.final_choice["center_original"]
            chosen_sphere = create_sphere(center_original, [1.0, 1.0, 0.0], radius=0.035) 
            self.geometries_to_draw.append(chosen_sphere)
        else:
            print("[CẢNH BÁO] Không có lựa chọn cuối cùng nào để vẽ quả cầu VÀNG.")


    def visualize(self):
        """Hiển thị tất cả các đối tượng."""
        
        # 1. Tạo các hộp config
        self._create_visualization_geometries()
        
        # 2. Thêm các đường bao prism (màu đỏ)
        self.geometries_to_draw.extend(self.prisms_for_drawing)

        # 3. Thêm quả cầu mục tiêu CSV (màu xanh lá lớn)
        target_sphere = create_sphere(self.target_coords_original, [0.0, 1.0, 0.0], radius=0.035)
        self.geometries_to_draw.append(target_sphere)
        
        print(f"\nĐang hiển thị {self.file_name}. Hãy đóng cửa sổ để tiếp tục...")
        
        window_title = (
            f"Tệp: {self.file_name} | "
            f"{self.num_valid_clusters} cụm | "
            f"{len(self.candidates)} ứng viên (YOLO Seg)"
        )
        
        o3d.visualization.draw_geometries(
            self.geometries_to_draw,
            window_name=window_title,
            width=self.config.WIDTH,
            height=self.config.HEIGHT
        )

# =============================================================================
# --- MAIN & HELPER FUNCTIONS ---
# =============================================================================

def load_labels_data(csv_path: str) -> List[List[float]]:
    """Tải dữ liệu label từ file CSV."""
    labels_list = []
    if not os.path.exists(csv_path):
        print(f"[CẢNH BÁO] Không tìm thấy file labels: {csv_path}")
        return []

    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                print(f"Đã đọc header CSV: {header}")
            except StopIteration:
                print(f"Lỗi: File CSV {csv_path} rỗng.")
                return []
            
            try:
                name_idx = header.index('file_name')
                x_idx = header.index('x')
                y_idx = header.index('y')
                z_idx = header.index('z')
            except ValueError:
                name_idx, x_idx, y_idx, z_idx = 0, 1, 2, 3

            for i, row in enumerate(reader):
                if len(row) < 4:
                    continue
                try:
                    coords = [float(row[x_idx]), float(row[y_idx]), float(row[z_idx])]
                    labels_list.append(coords)
                except (ValueError, IndexError):
                    pass
                    
        print(f"Đã tải thành công {len(labels_list)} labels từ CSV.")
        return labels_list

    except Exception as e:
        print(f"Lỗi nghiêm trọng khi đọc file CSV {csv_path}: {e}")
        return []


def main():
    """Hàm chính đã được cập nhật với model YOLO."""
    
    print("--- Bắt đầu chạy pipeline xử lý PCD (Logic Phân Vùng YOLOv8) ---")
    
    # 1. Khởi tạo cấu hình
    config = PipelineConfig()
    
    # 2. Tải mô hình YOLO (THAY ĐỔI)
    print(f"Đang tải mô hình YOLO từ '{config.YOLO_MODEL_PATH}'...")
    model_2d = None
    try:
        model_2d = YOLO(config.YOLO_MODEL_PATH)
        print("Tải mô hình YOLO thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình YOLO: {e}")
        print("[CẢNH BÁO] Không tải được model, sẽ tiếp tục chỉ với xử lý 3D.")

    # 3. Tải dữ liệu từ CSV
    labels_file_path = "./data/labels.csv" 
    labels_data = load_labels_data(labels_file_path)
    
    current_file_num = 0
    
    # 5. Bắt đầu vòng lặp tương tác
    while True:
        try:
            os.system('cls' if os.name == 'nt' else 'clear')

            test_file_base = f"{current_file_num:04d}"
            
            target_index = current_file_num
            if labels_data and 0 <= target_index < len(labels_data):
                test_coords = labels_data[target_index]
            else:
                test_coords = [0.0, 0.0, 1.0]
            
            print("="*60)
            print(f" SẮP CHẠY: {test_file_base}.png | Tọa độ CSV (Index {target_index}): {test_coords}")
            print("="*60)

            processor = PointCloudProcessor(
                config=config,
                file_base_name=test_file_base,
                target_coords=test_coords,
                model=model_2d
            )
            
            processor.run_pipeline()
            processor.visualize()
            
            print("\n" + "="*60)
            print("ĐÃ XỬ LÝ HOÀN TẤT. CHỜ PHÍM TIẾP THEO...")
            print(f"   File vừa chạy: {test_file_base}")
            print("   Bấm (->) để qua file tiếp, (<-) để lùi, (Q) hoặc (ESC) để thoát.")
            print("="*60)

            while True:
                event = keyboard.read_event(suppress=True)
                if event.event_type == keyboard.KEY_DOWN:
                    key = event.name
                    if key == 'right':
                        current_file_num += 1
                        break
                    elif key == 'left':
                        current_file_num -= 1
                        if current_file_num < 0: current_file_num = 0
                        break
                    elif key == 'q' or key == 'esc':
                        print("Đang thoát...")
                        return
            
        except KeyboardInterrupt:
            print("\nĐã nhận Ctrl+C. Đang thoát...")
            break
        except Exception as e:
            print(f"Lỗi nghiêm trọng trong vòng lặp chính: {e}")
            import traceback
            traceback.print_exc()
            return


if __name__ == "__main__":
    main()