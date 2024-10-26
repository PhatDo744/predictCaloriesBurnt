import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn import metrics

warnings.filterwarnings("ignore")

# Hàm tính Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Hàm tính Mean Absolute Error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Hàm tính R-squared
def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)  # Tổng bình phương sai số
    ss_residual = np.sum((y_true - y_pred) ** 2)         # Tổng bình phương sai số còn lại
    return 1 - (ss_residual / ss_total)

# Hàm tính MSE cho một phân chia
def mse_split(left_y, right_y):
    return (mse(left_y, np.mean(left_y)) * len(left_y) + 
            mse(right_y, np.mean(right_y)) * len(right_y)) / (len(left_y) + len(right_y))



class DataUtils:
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2
        self.data = None

    def load_and_merge_data(self):
        df1 = pd.read_csv(self.file1) 
        df2 = pd.read_csv(self.file2) 
        self.data = pd.merge(df1, df2, on="User_ID")  # Ghép 2 bảng df1 và df2 thành 1 bảng
        
    def preprocess_data(self):
        # Chuẩn hóa dữ liệu giới tính
        self.data.replace({'male': 0, 'female': 1}, inplace=True)

        # Lấy các cột đầu vào và biến mục tiêu
        features = self.data.drop(['User_ID', 'Calories'], axis=1)
        target = self.data['Calories'].values
        return features, target


class MyLinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        one = np.ones((X.shape[0], 1))
        Xbar = np.concatenate((one, X), axis=1)  # Ghép cột hệ số chặn
        A = np.dot(Xbar.T, Xbar)  # X^T.X
        b = np.dot(Xbar.T, y)  # X^T.y
        self.coefficients = np.dot(np.linalg.pinv(A), b)  # Tính pseudo-inverse

    def predict(self, X):
        one = np.ones((X.shape[0], 1))
        Xbar = np.concatenate((one, X), axis=1)  # Ghép cột hệ số chặn
        return np.dot(Xbar, self.coefficients)

    def evaluate(self, y_true, y_pred):
        r2 = metrics.r2_score(y_true, y_pred)
        return r2


# Hàm tìm split tốt nhất cho dữ liệu
def find_best_split(X, y):
    best_mse = float('inf')
    best_index = None
    best_value = None
    n_features = X.shape[1]

    for feature_index in range(n_features):
        values = np.unique(X[:, feature_index])
        
        for value in values:
            left_mask = X[:, feature_index] <= value
            right_mask = X[:, feature_index] > value
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            left_y = y[left_mask]
            right_y = y[right_mask]
            
            current_mse = mse_split(left_y, right_y)
            
            if current_mse < best_mse:
                best_mse = current_mse
                best_index = feature_index
                best_value = value

    return best_index, best_value

# Định nghĩa lớp cây hồi quy
class TreeNode:
    def __init__(self, feature_index=None, value=None, left=None, right=None):
        self.feature_index = feature_index
        self.value = value
        self.left = left
        self.right = right

class CART_DecisionTreeRegressor:
    def __init__(self, max_depth=15):
        self.max_depth = max_depth
        self.root = None

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) == 0:
            return TreeNode(value=np.mean(y))

        feature_index, split_value = find_best_split(X, y)
        
        if feature_index is None:
            return TreeNode(value=np.mean(y))
        
        left_mask = X[:, feature_index] <= split_value
        right_mask = X[:, feature_index] > split_value

        left_node = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature_index=feature_index, value=split_value, left=left_node, right=right_node)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_single(self, tree, x):
        if tree.left is None and tree.right is None:
            return tree.value
        if x[tree.feature_index] <= tree.value:
            return self.predict_single(tree.left, x)
        else:
            return self.predict_single(tree.right, x)

    def predict(self, X):
        return np.array([self.predict_single(self.root, x) for x in X])

# Định nghĩa lớp Random Forest Regressor
class RandomForestRegressor:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.forest = []

    def random_features(self, X):
        total_features = X.shape[1]
        num_features = min(self.max_features or total_features, total_features)
        return np.random.choice(range(total_features), size=num_features, replace=False)

    def bootstrap_sample(self, X, y):
        indices = np.random.randint(0, len(X), len(X))
        return X[indices], y[indices]

    def fit(self, X, y):
        X = X.values  # Chuyển đổi DataFrame thành numpy array
        y = np.array(y)
        for _ in range(self.n_trees):
            sample_X, sample_y = self.bootstrap_sample(X, y)
            selected_features = self.random_features(sample_X)
            sample_X = sample_X[:, selected_features]
            tree = CART_DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(sample_X, sample_y)
            self.forest.append((tree, selected_features))

    def predict(self, X):
        X = np.array(X)  # Chuyển đổi DataFrame thành numpy array nếu cần
        predictions = []
        for tree, features in self.forest:
            tree_predictions = tree.predict(X[:, features])
            predictions.append(tree_predictions)
        return np.mean(predictions, axis=0)


if __name__ == "__main__":
    # Khai báo và xử lý dữ liệu
    data_utils = DataUtils('calories.csv', 'exercise.csv')
    data_utils.load_and_merge_data()
    features, target = data_utils.preprocess_data()

    # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=22)

    # Tạo mô hình hồi quy tuyến tính và huấn luyện
    rg_model = MyLinearRegression()
    rg_model.fit(X_train.values, Y_train)

    # Dự đoán calories tiêu thụ
    y_pred_rg = rg_model.predict(X_val.values)

    # Đánh giá mô hình
    mse_value_rf = mse(Y_val, y_pred_rg)
    mae_value_rf = mae(Y_val, y_pred_rg)
    r2_value_rf = r2_score(Y_val, y_pred_rg)

    print(f'Mean Squared Error (Linear Regression): {mse_value_rf}')
    print(f'Mean Absolute Error (Linear Regression): {mae_value_rf}')
    print(f'R-squared (Linear Regression): {r2_value_rf}')


    # Tạo đối tượng RandomForestRegressor
    rf_model = RandomForestRegressor(n_trees=10, max_depth=15)

    # Huấn luyện mô hình
    rf_model.fit(X_train, Y_train)

    # Dự đoán trên tập kiểm tra
    Y_pred_rf = rf_model.predict(X_val)

    # Đánh giá mô hình
    mse_value_rf = mse(Y_val, Y_pred_rf)
    mae_value_rf = mae(Y_val, Y_pred_rf)
    r2_value_rf = r2_score(Y_val, Y_pred_rf)

    print(f'Mean Squared Error (Random Forest Regression): {mse_value_rf}')
    print(f'Mean Absolute Error (Random Forest Regression): {mae_value_rf}')
    print(f'R-squared (Random Forest Regression): {r2_value_rf}')

    # Tạo đối tượng CART Decision Tree Regressor
    cart_model = CART_DecisionTreeRegressor(max_depth=15)

    # Huấn luyện mô hình
    cart_model.fit(X_train.values, Y_train)

    # Dự đoán trên tập kiểm tra
    Y_pred_cart = cart_model.predict(X_val.values)

    # Đánh giá mô hình CART
    mse_value_cart = mse(Y_val, Y_pred_cart)
    mae_value_cart = mae(Y_val, Y_pred_cart)
    r2_value_cart = r2_score(Y_val, Y_pred_cart)

    print(f'Mean Squared Error (CART Decision Tree): {mse_value_cart}')
    print(f'Mean Absolute Error (CART Decision Tree): {mae_value_cart}')
    print(f'R-squared (CART Decision Tree): {r2_value_cart}')

    # # Vẽ biểu đồ so sánh giá trị thực và giá trị dự đoán của LinearRegression
    # plt.figure(figsize=(10, 6))
    # plt.scatter(Y_val, y_pred_rg, color='blue', label='Dự đoán', alpha=0.6)
    # plt.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], color='red', label='Giá trị thực', linestyle='--')
    # plt.title('So sánh Giá trị Thực và Giá trị Dự Đoán (LinearRegression)')
    # plt.xlabel('Giá trị Thực (Calories)')
    # plt.ylabel('Giá trị Dự Đoán (Calories)')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # # Biểu đồ so sánh giá trị thực và giá trị dự đoán của CART Decision Tree
    # plt.figure(figsize=(10, 6))
    # plt.scatter(Y_val, Y_pred_cart, color='green', label='Dự đoán (CART)', alpha=0.6)
    # plt.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], color='red', label='Giá trị thực', linestyle='--')
    # plt.title('So sánh Giá trị Thực và Giá trị Dự Đoán (CART Decision Tree)')
    # plt.xlabel('Giá trị Thực (Calories)')
    # plt.ylabel('Giá trị Dự Đoán (Calories)')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # # Biểu đồ so sánh giá trị thực và giá trị dự đoán của Random Forest
    # plt.figure(figsize=(10, 6))
    # plt.scatter(Y_val, Y_pred_rf, color='Blue')
    # plt.xlabel('Giá trị thực')
    # plt.ylabel('Giá trị dự đoán')
    # plt.title('So sánh giá trị thực và giá trị dự đoán (Random Forest)')
    # plt.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], 'k--', lw=2)
    # plt.show()
    # print("done!")

    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    # Giả sử X_train, Y_train, X_val, Y_val đã được định nghĩa

    models = [
        LinearRegression(),
        DecisionTreeRegressor(random_state=22),
        RandomForestRegressor(random_state=22)
    ]

    model_names = ['Linear Regression', 'Decision Tree', 'Random Forest']

    # Hàm tính toán và in các chỉ số đánh giá
    def evaluate_model(model, X_train, Y_train, X_val, Y_val):
        model.fit(X_train, Y_train)
        
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        
        mse_md = mse(Y_val, val_preds)
        mae_md = mae(Y_val, val_preds)
        r2 = r2_score(Y_val, val_preds)
        
        return train_preds, val_preds, mse_md, mae_md, r2

    # Lưu trữ kết quả
    results = []

    # Đánh giá từng mô hình
    for i, model in enumerate(models):
        train_preds, val_preds, mse_md, mae_md, r2 = evaluate_model(model, X_train, Y_train, X_val, Y_val)
        results.append({
            'model_name': model_names[i],
            'train_preds': train_preds,
            'val_preds': val_preds,
            'mse': mse_md,
            'mae': mae_md,
            'r2': r2
        })
        print(f'{model_names[i]}: MSE = {mse_md:.4f}, MAE = {mae_md:.4f}, R2 = {r2:.4f}')

    # # Vẽ biểu đồ so sánh giá trị thực và giá trị dự đoán
    # plt.figure(figsize=(18, 5))  # Kích thước khung hình

    # for i, result in enumerate(results):
    #     plt.subplot(1, 3, i + 1)  # 1 hàng, 3 cột, chỉ số thứ i + 1
    #     plt.scatter(Y_val, result['val_preds'], label=f'{result["model_name"]} Predictions', alpha=0.6)
    #     plt.plot(Y_val, Y_val, 'r-', label='True Values')  # Đường thẳng biểu diễn giá trị thực
    #     plt.xlabel('True Values')
    #     plt.ylabel('Predicted Values')
    #     plt.title(f'{result["model_name"]} Predictions')
    #     plt.legend()
    #     plt.grid()

    # plt.tight_layout()  # Căn chỉnh để không bị chồng chéo
    # plt.show()
    
    
    # Đoạn mã vẽ biểu đồ so sánh cho cả 6 mô hình
    plt.figure(figsize=(18, 10))  # Kích thước khung hình

    # Danh sách các mô hình tự triển khai và mô hình sử dụng thư viện
    all_models = models + [
        MyLinearRegression(),  # Mô hình hồi quy tuyến tính tự triển khai
        CART_DecisionTreeRegressor(max_depth=15),  # Mô hình CART tự triển khai
        RandomForestRegressor(n_estimators=10, max_depth=15)  # Mô hình Random Forest tự triển khai
    ]
    all_model_names = model_names + ['My Linear Regression', 'My CART', "Phat's Random Forest"]

    # Màu sắc cho từng nhóm mô hình
    colors = {
        'Linear Regression': 'blue',
        'Decision Tree': 'green',
        'Random Forest': 'orange',
        'My Linear Regression': 'blue',
        'My CART': 'green',
        'My Random Forest': 'orange',
        "Phat's Random Forest": 'orange',  # Thêm màu cho mô hình của bạn
    }

    # Đánh giá từng mô hình
    results_all = []
    for i, model in enumerate(all_models):
        if i < 3:  # Mô hình sử dụng thư viện
            train_preds, val_preds, mse_md, mae_md, r2 = evaluate_model(model, X_train, Y_train, X_val, Y_val)
        else:  # Mô hình tự triển khai
            if i == 3:
                model.fit(X_train.values, Y_train)
                val_preds = model.predict(X_val.values)
            elif i == 4:
                model.fit(X_train.values, Y_train)
                val_preds = model.predict(X_val.values)
            else:
                model.fit(X_train, Y_train)
                val_preds = model.predict(X_val)

        results_all.append(val_preds)
        
        # Vẽ biểu đồ cho từng mô hình với màu sắc đã xác định
        plt.subplot(2, 3, i + 1)  # 2 hàng, 3 cột, chỉ số thứ i + 1
        plt.scatter(Y_val, val_preds, label=f'{all_model_names[i]} Predictions', alpha=0.6, color=colors[all_model_names[i]])
        plt.plot(Y_val, Y_val, 'r-', label='True Values')  # Đường thẳng biểu diễn giá trị thực
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{all_model_names[i]} Predictions')
        plt.legend()
        plt.grid()

    plt.tight_layout()  # Căn chỉnh để không bị chồng chéo
    plt.show()
