# ============================================================================
# iPhone Sales Data Mining Analysis
# ============================================================================

# 加载必要的库
library(dplyr)
library(tidyr)
library(ggplot2)
library(arules)
library(arulesViz)
library(cluster)
library(factoextra)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(lubridate)
library(scales)
library(gridExtra)

# ============================================================================
# 数据加载与预处理
# ============================================================================

# 读取数据
data <- read.csv("iphone_sales_dataset.csv", stringsAsFactors = FALSE)

# 数据概览
str(data)
summary(data)
head(data)

# 数据预处理
data$Sale_Date <- as.Date(data$Sale_Date)
data$Month <- format(data$Sale_Date, "%Y-%m")
data$Total_Revenue <- data$Quantity * data$Price

# 将分类变量转换为因子
data$Country <- as.factor(data$Country)
data$iPhone_Model <- as.factor(data$iPhone_Model)
data$Storage <- as.factor(data$Storage)
data$Color <- as.factor(data$Color)
data$Payment_Method <- as.factor(data$Payment_Method)

# ============================================================================
# 任务1: 描述性分析 (Descriptive Analysis)
# ============================================================================

cat("\n========== 任务1: 描述性分析 ==========\n")

# 1.1 各国家/地区的销售分布
country_stats <- data %>%
  group_by(Country) %>%
  summarise(
    Total_Orders = n(),
    Total_Quantity = sum(Quantity),
    Total_Revenue = sum(Total_Revenue),
    Avg_Price = mean(Price),
    Avg_Quantity = mean(Quantity)
  ) %>%
  arrange(desc(Total_Revenue))

print("各国家/地区销售统计:")
print(country_stats)

# 1.2 各 iPhone 型号的销量和销售额排名
model_stats <- data %>%
  group_by(iPhone_Model) %>%
  summarise(
    Total_Orders = n(),
    Total_Quantity = sum(Quantity),
    Total_Revenue = sum(Total_Revenue),
    Avg_Price = mean(Price)
  ) %>%
  arrange(desc(Total_Quantity))

print("\niPhone型号销售统计:")
print(model_stats)

# 1.3 不同存储容量的偏好分析
storage_stats <- data %>%
  group_by(Storage) %>%
  summarise(
    Total_Orders = n(),
    Total_Quantity = sum(Quantity),
    Total_Revenue = sum(Total_Revenue),
    Percentage = n() / nrow(data) * 100
  ) %>%
  arrange(desc(Total_Orders))

print("\n存储容量偏好分析:")
print(storage_stats)

# 1.4 颜色偏好分析
color_stats <- data %>%
  group_by(Color) %>%
  summarise(
    Total_Orders = n(),
    Total_Quantity = sum(Quantity),
    Percentage = n() / nrow(data) * 100
  ) %>%
  arrange(desc(Total_Orders))

print("\n颜色偏好分析:")
print(color_stats)

# 1.5 支付方式使用频率分布
payment_stats <- data %>%
  group_by(Payment_Method) %>%
  summarise(
    Total_Orders = n(),
    Total_Revenue = sum(Total_Revenue),
    Percentage = n() / nrow(data) * 100
  ) %>%
  arrange(desc(Total_Orders))

print("\n支付方式分布:")
print(payment_stats)

# 1.6 月度销售趋势分析
monthly_stats <- data %>%
  group_by(Month) %>%
  summarise(
    Total_Orders = n(),
    Total_Quantity = sum(Quantity),
    Total_Revenue = sum(Total_Revenue),
    Avg_Price = mean(Price)
  ) %>%
  arrange(Month)

print("\n月度销售趋势:")
print(monthly_stats)

# 1.7 基本统计量
cat("\n价格统计:\n")
summary(data$Price)
cat("\n数量统计:\n")
summary(data$Quantity)
cat("\n总营收统计:\n")
summary(data$Total_Revenue)


# ============================================================================
# 任务2: 关联规则挖掘 (Association Rule Mining)
# ============================================================================

cat("\n\n========== 任务2: 关联规则挖掘 ==========\n")

# 2.1 准备交易数据 - 创建商品组合
transactions_data <- data %>%
  select(Country, iPhone_Model, Storage, Color, Payment_Method)

# 将数据转换为交易格式
transaction_list <- lapply(1:nrow(transactions_data), function(i) {
  paste(names(transactions_data), unlist(transactions_data[i, ]), sep = "=")
})

# 创建交易数据集
trans <- as(transaction_list, "transactions")

# 查看交易数据摘要
summary(trans)

# 2.2 使用 Apriori 算法挖掘关联规则
# 设置最小支持度和最小置信度
rules <- apriori(trans, 
                 parameter = list(support = 0.05, 
                                  confidence = 0.3, 
                                  minlen = 2,
                                  maxlen = 5))

# 查看规则摘要
summary(rules)

# 2.3 按置信度排序，查看前20条规则
rules_conf <- sort(rules, by = "confidence", decreasing = TRUE)
print("\n按置信度排序的前20条规则:")
inspect(head(rules_conf, 20))

# 2.4 按提升度排序，查看前20条规则
rules_lift <- sort(rules, by = "lift", decreasing = TRUE)
print("\n按提升度排序的前20条规则:")
inspect(head(rules_lift, 20))

# 2.5 按支持度排序，查看前20条规则
rules_supp <- sort(rules, by = "support", decreasing = TRUE)
print("\n按支持度排序的前20条规则:")
inspect(head(rules_supp, 20))

# 2.6 挖掘特定规则 - 预测支付方式
payment_rules <- apriori(trans, 
                         parameter = list(support = 0.03, 
                                          confidence = 0.2),
                         appearance = list(rhs = c("Payment_Method=Cash",
                                                    "Payment_Method=Credit Card",
                                                    "Payment_Method=Debit Card",
                                                    "Payment_Method=PayPal"),
                                           default = "lhs"))

print("\n预测支付方式的规则:")
inspect(head(sort(payment_rules, by = "confidence"), 15))

# 2.7 挖掘国家与产品配置的关联
country_rules <- apriori(trans,
                         parameter = list(support = 0.03,
                                          confidence = 0.2),
                         appearance = list(lhs = c("Country=USA",
                                                    "Country=Canada",
                                                    "Country=UK",
                                                    "Country=Germany",
                                                    "Country=UAE",
                                                    "Country=Pakistan"),
                                           default = "rhs"))

print("\n国家与产品配置关联规则:")
inspect(head(sort(country_rules, by = "lift"), 15))

# 2.8 将规则转换为数据框以便进一步分析
rules_df <- as(rules, "dataFrame")
print("\n关联规则数据框(前10条):")
head(rules_df, 10)


# ============================================================================
# 任务3: 聚类分析 (Clustering)
# ============================================================================

cat("\n\n========== 任务3: 聚类分析 ==========\n")

# 3.1 客户购买行为聚类
# 聚合客户级别的数据
customer_data <- data %>%
  group_by(Customer_Name) %>%
  summarise(
    Total_Orders = n(),
    Total_Quantity = sum(Quantity),
    Total_Revenue = sum(Total_Revenue),
    Avg_Price = mean(Price),
    Max_Quantity = max(Quantity),
    Unique_Models = n_distinct(iPhone_Model),
    Unique_Countries = n_distinct(Country)
  )

# 准备聚类特征
cluster_features <- customer_data %>%
  select(Total_Orders, Total_Quantity, Total_Revenue, Avg_Price, Max_Quantity, Unique_Models)

# 数据标准化
cluster_scaled <- scale(cluster_features)

# 3.2 使用肘部法则确定最佳聚类数
wss <- sapply(1:10, function(k) {
  kmeans(cluster_scaled, k, nstart = 25, iter.max = 100)$tot.withinss
})

# 绘制肘部图
plot(1:10, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "聚类数量 K",
     ylab = "组内平方和 (WSS)",
     main = "肘部法则确定最佳聚类数")

# 3.3 K-Means 聚类 (假设选择3个聚类)
set.seed(123)
k <- 3
kmeans_result <- kmeans(cluster_scaled, centers = k, nstart = 25, iter.max = 100)

# 查看聚类结果
print("\nK-Means 聚类结果:")
print(kmeans_result$centers)
print(table(kmeans_result$cluster))

# 将聚类结果添加到客户数据
customer_data$Cluster <- as.factor(kmeans_result$cluster)

# 3.4 聚类特征分析
cluster_profile <- customer_data %>%
  group_by(Cluster) %>%
  summarise(
    Customer_Count = n(),
    Avg_Orders = mean(Total_Orders),
    Avg_Quantity = mean(Total_Quantity),
    Avg_Revenue = mean(Total_Revenue),
    Avg_Price = mean(Avg_Price),
    Total_Revenue_Sum = sum(Total_Revenue)
  )

print("\n聚类特征分析:")
print(cluster_profile)

# 3.5 可视化聚类结果
# PCA 降维可视化
pca_result <- prcomp(cluster_scaled, scale. = FALSE)
pca_data <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  Cluster = customer_data$Cluster
)

ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "客户聚类分析 (PCA降维)",
       x = "主成分1",
       y = "主成分2") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 3.6 国家/地区市场聚类
country_cluster <- data %>%
  group_by(Country) %>%
  summarise(
    Avg_Price = mean(Price),
    Avg_Quantity = mean(Quantity),
    Total_Revenue = sum(Total_Revenue),
    Order_Count = n(),
    Model_Diversity = n_distinct(iPhone_Model),
    Payment_Diversity = n_distinct(Payment_Method)
  )

country_scaled <- scale(country_cluster[, -1])

# 层次聚类
dist_matrix <- dist(country_scaled, method = "euclidean")
hc <- hclust(dist_matrix, method = "ward.D2")

# 绘制树状图
plot(hc, labels = country_cluster$Country,
     main = "国家/地区市场层次聚类",
     xlab = "国家",
     sub = "",
     cex = 0.8)
rect.hclust(hc, k = 3, border = "red")

# 3.7 产品配置聚类
product_data <- data %>%
  group_by(iPhone_Model, Storage, Color) %>%
  summarise(
    Total_Sales = sum(Quantity),
    Total_Revenue = sum(Total_Revenue),
    Avg_Price = mean(Price),
    Order_Count = n(),
    .groups = "drop"
  )

product_scaled <- scale(product_data[, c("Total_Sales", "Total_Revenue", "Avg_Price", "Order_Count")])

set.seed(123)
product_kmeans <- kmeans(product_scaled, centers = 4, nstart = 25)

product_data$Product_Cluster <- as.factor(product_kmeans$cluster)

print("\n产品聚类中心:")
print(product_kmeans$centers)

# 查看各聚类的产品
print("\n各聚类中的产品:")
product_data %>%
  arrange(Product_Cluster, desc(Total_Sales)) %>%
  select(Product_Cluster, iPhone_Model, Storage, Color, Total_Sales, Total_Revenue)


# ============================================================================
# 任务4: 分类与预测 (Classification & Prediction)
# ============================================================================

cat("\n\n========== 任务4: 分类与预测 ==========\n")

# 4.1 预测支付方式分类
# 准备数据
classification_data <- data %>%
  mutate(
    Model_Num = as.numeric(gsub("iPhone ", "", gsub(" Pro Max", "5", gsub(" Pro", "4", iPhone_Model)))),
    Storage_Num = as.numeric(gsub("GB", "", Storage)),
    Month_Num = as.numeric(format(Sale_Date, "%m"))
  )

# 划分训练集和测试集
set.seed(123)
train_index <- createDataPartition(classification_data$Payment_Method, p = 0.7, list = FALSE)
train_data <- classification_data[train_index, ]
test_data <- classification_data[-train_index, ]

# 4.2 决策树分类 - 预测支付方式
tree_model <- rpart(Payment_Method ~ Country + Model_Num + Storage_Num + 
                      Color + Quantity + Price + Month_Num,
                    data = train_data,
                    method = "class",
                    control = rpart.control(minsplit = 5, cp = 0.01))

# 可视化决策树
rpart.plot(tree_model, main = "支付方式预测决策树", cex = 0.8)

# 预测
tree_pred <- predict(tree_model, test_data, type = "class")

# 评估模型
confusion_matrix_tree <- confusionMatrix(tree_pred, test_data$Payment_Method)
print("\n决策树混淆矩阵:")
print(confusion_matrix_tree)

cat("\n决策树准确率:", confusion_matrix_tree$overall["Accuracy"], "\n")

# 4.3 随机森林分类
rf_model <- randomForest(Payment_Method ~ Country + Model_Num + Storage_Num + 
                           Color + Quantity + Price + Month_Num,
                         data = train_data,
                         ntree = 100,
                         mtry = 3,
                         importance = TRUE)

# 预测
rf_pred <- predict(rf_model, test_data)

# 评估模型
confusion_matrix_rf <- confusionMatrix(rf_pred, test_data$Payment_Method)
print("\n随机森林混淆矩阵:")
print(confusion_matrix_rf)

cat("\n随机森林准确率:", confusion_matrix_rf$overall["Accuracy"], "\n")

# 4.4 变量重要性分析
varImpPlot(rf_model, main = "随机森林变量重要性")

# 4.5 预测 iPhone 型号偏好
# 将型号分为高端和标准
classification_data$Model_Category <- ifelse(
  grepl("Pro", classification_data$iPhone_Model),
  "Pro/Pro Max",
  "Standard"
)
classification_data$Model_Category <- as.factor(classification_data$Model_Category)

# 重新划分数据
train_index2 <- createDataPartition(classification_data$Model_Category, p = 0.7, list = FALSE)
train_data2 <- classification_data[train_index2, ]
test_data2 <- classification_data[-train_index2, ]

# 构建预测模型
model_tree <- rpart(Model_Category ~ Country + Storage_Num + Color + 
                      Quantity + Price + Payment_Method + Month_Num,
                    data = train_data2,
                    method = "class")

# 预测与评估
model_pred <- predict(model_tree, test_data2, type = "class")
confusion_matrix_model <- confusionMatrix(model_pred, test_data2$Model_Category)

print("\niPhone型号预测混淆矩阵:")
print(confusion_matrix_model)

cat("\niPhone型号预测准确率:", confusion_matrix_model$overall["Accuracy"], "\n")

# 4.6 价格预测（回归）
# 准备回归数据
regression_data <- data %>%
  mutate(
    Model_Num = as.numeric(gsub("iPhone ", "", gsub(" Pro Max", "5", gsub(" Pro", "4", iPhone_Model)))),
    Storage_Num = as.numeric(gsub("GB", "", Storage)),
    Month_Num = as.numeric(format(Sale_Date, "%m"))
  )

# 划分训练集和测试集
set.seed(123)
train_idx <- createDataPartition(regression_data$Price, p = 0.7, list = FALSE)
train_reg <- regression_data[train_idx, ]
test_reg <- regression_data[-train_idx, ]

# 线性回归模型
lm_model <- lm(Price ~ Country + Model_Num + Storage_Num + Color + 
                 Quantity + Payment_Method + Month_Num,
               data = train_reg)

# 模型摘要
summary(lm_model)

# 预测
lm_pred <- predict(lm_model, test_reg)

# 评估回归模型
rmse <- sqrt(mean((lm_pred - test_reg$Price)^2))
mae <- mean(abs(lm_pred - test_reg$Price))
r_squared <- cor(lm_pred, test_reg$Price)^2

cat("\n线性回归模型评估:\n")
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("R-squared:", r_squared, "\n")

# 4.7 时间序列预测 - 月度销售额
monthly_revenue <- data %>%
  group_by(Month) %>%
  summarise(Total_Revenue = sum(Total_Revenue)) %>%
  arrange(Month)

# 创建时间序列对象
ts_revenue <- ts(monthly_revenue$Total_Revenue, start = c(2025, 1), frequency = 12)

# 简单线性趋势模型
time_seq <- 1:length(ts_revenue)
trend_model <- lm(ts_revenue ~ time_seq)
summary(trend_model)

# 预测未来3个月
future_time <- (length(time_seq) + 1):(length(time_seq) + 3)
future_predictions <- predict(trend_model, newdata = data.frame(time_seq = future_time))

cat("\n未来3个月销售额预测:\n")
print(future_predictions)


# ============================================================================
# 任务5: 异常检测 (Anomaly Detection)
# ============================================================================

cat("\n\n========== 任务5: 异常检测 ==========\n")

# 5.1 价格异常检测 - IQR 方法
Q1 <- quantile(data$Price, 0.25)
Q3 <- quantile(data$Price, 0.75)
IQR_val <- Q3 - Q1

lower_bound <- Q1 - 1.5 * IQR_val
upper_bound <- Q3 + 1.5 * IQR_val

price_anomalies <- data %>%
  filter(Price < lower_bound | Price > upper_bound) %>%
  arrange(desc(Price))

cat("\n价格异常检测 (IQR方法):\n")
cat("下界:", lower_bound, "\n")
cat("上界:", upper_bound, "\n")
cat("异常订单数量:", nrow(price_anomalies), "\n")

if(nrow(price_anomalies) > 0) {
  print("\n异常价格订单:")
  print(price_anomalies %>% select(Order_ID, Country, iPhone_Model, Price, Quantity, Total_Revenue))
}

# 5.2 价格异常检测 - Z-Score 方法
price_mean <- mean(data$Price)
price_sd <- sd(data$Price)

data$Price_ZScore <- abs((data$Price - price_mean) / price_sd)

price_zscore_anomalies <- data %>%
  filter(Price_ZScore > 2) %>%
  arrange(desc(Price_ZScore))

cat("\n价格异常检测 (Z-Score方法, 阈值=2):\n")
cat("均值:", price_mean, "\n")
cat("标准差:", price_sd, "\n")
cat("异常订单数量:", nrow(price_zscore_anomalies), "\n")

if(nrow(price_zscore_anomalies) > 0) {
  print("\nZ-Score异常价格订单:")
  print(price_zscore_anomalies %>% select(Order_ID, Country, iPhone_Model, Price, Price_ZScore))
}

# 5.3 购买数量异常检测
quantity_stats <- data %>%
  summarise(
    Q1 = quantile(Quantity, 0.25),
    Q3 = quantile(Quantity, 0.75),
    IQR = Q3 - Q1,
    Lower = Q1 - 1.5 * IQR,
    Upper = Q3 + 1.5 * IQR
  )

quantity_anomalies <- data %>%
  filter(Quantity < quantity_stats$Lower | Quantity > quantity_stats$Upper) %>%
  arrange(desc(Quantity))

cat("\n购买数量异常检测 (IQR方法):\n")
cat("下界:", quantity_stats$Lower, "\n")
cat("上界:", quantity_stats$Upper, "\n")
cat("异常订单数量:", nrow(quantity_anomalies), "\n")

if(nrow(quantity_anomalies) > 0) {
  print("\n异常数量订单:")
  print(quantity_anomalies %>% select(Order_ID, Country, iPhone_Model, Quantity, Price, Total_Revenue))
}

# 5.4 总营收异常检测
revenue_mean <- mean(data$Total_Revenue)
revenue_sd <- sd(data$Total_Revenue)

data$Revenue_ZScore <- abs((data$Total_Revenue - revenue_mean) / revenue_sd)

revenue_anomalies <- data %>%
  filter(Revenue_ZScore > 2.5) %>%
  arrange(desc(Revenue_ZScore))

cat("\n总营收异常检测 (Z-Score方法, 阈值=2.5):\n")
cat("均值:", revenue_mean, "\n")
cat("标准差:", revenue_sd, "\n")
cat("异常订单数量:", nrow(revenue_anomalies), "\n")

if(nrow(revenue_anomalies) > 0) {
  print("\n异常营收订单:")
  print(revenue_anomalies %>% select(Order_ID, Country, iPhone_Model, Quantity, Price, Total_Revenue, Revenue_ZScore))
}

# 5.5 多变量异常检测 - 马氏距离
library(MASS)

anomaly_features <- data %>%
  select(Price, Quantity, Total_Revenue)

anomaly_scaled <- scale(anomaly_features)

mahal_dist <- mahalanobis(anomaly_scaled, 
                          colMeans(anomaly_scaled), 
                          cov(anomaly_scaled))

data$Mahalanobis_Dist <- mahal_dist

threshold <- qchisq(0.975, df = ncol(anomaly_scaled))

multivariate_anomalies <- data %>%
  filter(Mahalanobis_Dist > threshold) %>%
  arrange(desc(Mahalanobis_Dist))

cat("\n多变量异常检测 (马氏距离, 97.5%置信度):\n")
cat("阈值:", threshold, "\n")
cat("异常订单数量:", nrow(multivariate_anomalies), "\n")

if(nrow(multivariate_anomalies) > 0) {
  print("\n多变量异常订单:")
  print(multivariate_anomalies %>% select(Order_ID, Country, iPhone_Model, Price, Quantity, Total_Revenue, Mahalanobis_Dist))
}

# 5.6 按国家分组的异常检测
country_anomalies <- data %>%
  group_by(Country) %>%
  mutate(
    Country_Mean = mean(Price),
    Country_SD = sd(Price),
    Country_ZScore = abs((Price - Country_Mean) / Country_SD)
  ) %>%
  filter(Country_ZScore > 2) %>%
  arrange(Country, desc(Country_ZScore))

cat("\n按国家分组的异常检测 (Z-Score > 2):\n")
cat("异常订单数量:", nrow(country_anomalies), "\n")

if(nrow(country_anomalies) > 0) {
  print("\n各国家异常订单汇总:")
  print(country_anomalies %>% 
          group_by(Country) %>% 
          summarise(Anomaly_Count = n(), 
                    Avg_ZScore = mean(Country_ZScore)) %>%
          arrange(desc(Anomaly_Count)))
}

# 5.7 按型号分组的异常检测
model_anomalies <- data %>%
  group_by(iPhone_Model) %>%
  mutate(
    Model_Mean = mean(Price),
    Model_SD = sd(Price),
    Model_ZScore = abs((Price - Model_Mean) / Model_SD)
  ) %>%
  filter(Model_ZScore > 2) %>%
  arrange(iPhone_Model, desc(Model_ZScore))

cat("\n按型号分组的异常检测 (Z-Score > 2):\n")
cat("异常订单数量:", nrow(model_anomalies), "\n")

if(nrow(model_anomalies) > 0) {
  print("\n各型号异常订单汇总:")
  print(model_anomalies %>% 
          group_by(iPhone_Model) %>% 
          summarise(Anomaly_Count = n(), 
                    Avg_ZScore = mean(Model_ZScore)) %>%
          arrange(desc(Anomaly_Count)))
}

# 5.8 异常检测可视化
ggplot(data, aes(x = Price, y = Quantity)) +
  geom_point(aes(color = ifelse(Mahalanobis_Dist > threshold, "异常", "正常")), 
             alpha = 0.6, size = 2) +
  scale_color_manual(values = c("正常" = "steelblue", "异常" = "red")) +
  labs(title = "价格-数量散点图 (异常点标记)",
       x = "价格 ($)",
       y = "购买数量",
       color = "类别") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 5.9 异常检测汇总
cat("\n========== 异常检测汇总 ==========\n")
cat("价格异常 (IQR):", nrow(price_anomalies), "笔\n")
cat("价格异常 (Z-Score):", nrow(price_zscore_anomalies), "笔\n")
cat("数量异常:", nrow(quantity_anomalies), "笔\n")
cat("营收异常:", nrow(revenue_anomalies), "笔\n")
cat("多变量异常:", nrow(multivariate_anomalies), "笔\n")
cat("国家分组异常:", nrow(country_anomalies), "笔\n")
cat("型号分组异常:", nrow(model_anomalies), "笔\n")

all_anomaly_ids <- unique(c(
  price_anomalies$Order_ID,
  price_zscore_anomalies$Order_ID,
  quantity_anomalies$Order_ID,
  revenue_anomalies$Order_ID,
  multivariate_anomalies$Order_ID
))

cat("\n所有异常订单总数 (去重):", length(all_anomaly_ids), "笔\n")
cat("异常率:", round(length(all_anomaly_ids) / nrow(data) * 100, 2), "%\n")


# ============================================================================
# 任务6: 交叉分析 (Cross-tabulation)
# ============================================================================

cat("\n\n========== 任务6: 交叉分析 ==========\n")

# 6.1 国家 × iPhone型号 交叉表
country_model_table <- table(data$Country, data$iPhone_Model)
print("国家 × iPhone型号 交叉表:")
print(country_model_table)

# 带百分比的交叉表
country_model_prop <- prop.table(country_model_table, margin = 1) * 100
print("\n国家 × iPhone型号 百分比交叉表 (行百分比):")
print(round(country_model_prop, 1))

# 6.2 卡方检验 - 国家与型号是否独立
chi_test_country_model <- chisq.test(country_model_table)
print("\n卡方检验 - 国家与型号独立性:")
print(chi_test_country_model)

# 6.3 国家 × 存储容量 交叉表
country_storage_table <- table(data$Country, data$Storage)
print("\n国家 × 存储容量 交叉表:")
print(country_storage_table)

country_storage_prop <- prop.table(country_storage_table, margin = 1) * 100
print("\n国家 × 存储容量 百分比交叉表:")
print(round(country_storage_prop, 1))

# 6.4 国家 × 颜色 交叉表
country_color_table <- table(data$Country, data$Color)
print("\n国家 × 颜色 交叉表:")
print(country_color_table)

# 6.5 国家 × 支付方式 交叉表
country_payment_table <- table(data$Country, data$Payment_Method)
print("\n国家 × 支付方式 交叉表:")
print(country_payment_table)

country_payment_prop <- prop.table(country_payment_table, margin = 1) * 100
print("\n国家 × 支付方式 百分比交叉表:")
print(round(country_payment_prop, 1))

# 6.6 iPhone型号 × 存储容量 交叉表
model_storage_table <- table(data$iPhone_Model, data$Storage)
print("\niPhone型号 × 存储容量 交叉表:")
print(model_storage_table)

# 6.7 iPhone型号 × 颜色 交叉表
model_color_table <- table(data$iPhone_Model, data$Color)
print("\niPhone型号 × 颜色 交叉表:")
print(model_color_table)

# 6.8 月份 × 国家 销售趋势交叉表
month_country_table <- table(data$Month, data$Country)
print("\n月份 × 国家 销售趋势交叉表:")
print(month_country_table)

# 6.9 月份 × 型号 销售趋势交叉表
month_model_table <- table(data$Month, data$iPhone_Model)
print("\n月份 × 型号 销售趋势交叉表:")
print(month_model_table)

# 6.10 使用 xtabs 进行多维度交叉分析
# 国家 × 型号 × 支付方式
multi_table <- xtabs(~ Country + iPhone_Model + Payment_Method, data = data)
print("\n三维交叉表摘要 (国家 × 型号 × 支付方式):")
print(ftable(multi_table))

# 6.11 带聚合函数的交叉表
# 国家 × 型号的平均价格
avg_price_cross <- data %>%
  group_by(Country, iPhone_Model) %>%
  summarise(Avg_Price = mean(Price),
            Total_Sales = sum(Quantity),
            .groups = "drop") %>%
  pivot_wider(names_from = iPhone_Model, 
              values_from = Avg_Price,
              values_fill = 0)

print("\n国家 × 型号 平均价格交叉表:")
print(round(avg_price_cross, 2))

# 6.12 颜色 × 存储容量偏好分析
color_storage_table <- table(data$Color, data$Storage)
print("\n颜色 × 存储容量 交叉表:")
print(color_storage_table)

color_storage_prop <- prop.table(color_storage_table, margin = 1) * 100
print("\n颜色 × 存储容量 百分比交叉表:")
print(round(color_storage_prop, 1))


# ============================================================================
# 任务8: 可视化分析 (Visual Analytics)
# ============================================================================

cat("\n\n========== 任务8: 可视化分析 ==========\n")

# 8.1 各国家销售分布 - 柱状图
p1 <- ggplot(country_stats, aes(x = reorder(Country, -Total_Revenue), y = Total_Revenue, fill = Country)) +
  geom_bar(stat = "identity") +
  labs(title = "各国家/地区总销售额",
       x = "国家",
       y = "总销售额 ($)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5)) +
  geom_text(aes(label = paste0("$", format(Total_Revenue, big.mark = ","))), 
            vjust = -0.5, size = 3)

print(p1)

# 8.2 iPhone型号销量分布 - 饼图
p2 <- ggplot(model_stats, aes(x = "", y = Total_Quantity, fill = iPhone_Model)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  labs(title = "iPhone型号销量分布",
       fill = "型号") +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_text(aes(label = paste0(round(Total_Quantity/sum(Total_Quantity)*100, 1), "%")),
            position = position_stack(vjust = 0.5), size = 3)

print(p2)

# 8.3 颜色偏好 - 条形图
p3 <- ggplot(color_stats, aes(x = reorder(Color, -Total_Orders), y = Total_Orders, fill = Color)) +
  geom_bar(stat = "identity") +
  labs(title = "颜色偏好分布",
       x = "颜色",
       y = "订单数量") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "none")

print(p3)

# 8.4 支付方式分布 - 饼图
p4 <- ggplot(payment_stats, aes(x = "", y = Total_Orders, fill = Payment_Method)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  labs(title = "支付方式分布",
       fill = "支付方式") +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_text(aes(label = paste0(round(Percentage, 1), "%")),
            position = position_stack(vjust = 0.5), size = 3.5)

print(p4)

# 8.5 月度销售趋势 - 折线图
p5 <- ggplot(monthly_stats, aes(x = Month, y = Total_Revenue, group = 1)) +
  geom_line(color = "steelblue", size = 1.2) +
  geom_point(color = "steelblue", size = 3) +
  labs(title = "月度销售额趋势",
       x = "月份",
       y = "总销售额 ($)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5)) +
  geom_text(aes(label = paste0("$", format(Total_Revenue, big.mark = ","))),
            vjust = -1, size = 3)

print(p5)

# 8.6 国家 × 型号 热力图
heatmap_data <- data %>%
  group_by(Country, iPhone_Model) %>%
  summarise(Total_Sales = sum(Quantity), .groups = "drop")

p6 <- ggplot(heatmap_data, aes(x = iPhone_Model, y = Country, fill = Total_Sales)) +
  geom_tile() +
  geom_text(aes(label = Total_Sales), color = "white", size = 4) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "国家 × iPhone型号 销售热力图",
       x = "iPhone型号",
       y = "国家",
       fill = "销量") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5))

print(p6)

# 8.7 价格分布 - 箱线图
p7 <- ggplot(data, aes(x = iPhone_Model, y = Price, fill = iPhone_Model)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "各iPhone型号价格分布",
       x = "iPhone型号",
       y = "价格 ($)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5),
        legend.position = "none")

print(p7)

# 8.8 价格分布 - 直方图
p8 <- ggplot(data, aes(x = Price, fill = Country)) +
  geom_histogram(binwidth = 100, alpha = 0.7, position = "identity") +
  labs(title = "价格分布直方图",
       x = "价格 ($)",
       y = "频数",
       fill = "国家") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

print(p8)

# 8.9 存储容量偏好 - 堆叠条形图
storage_country_data <- data %>%
  group_by(Country, Storage) %>%
  summarise(Count = n(), .groups = "drop")

p9 <- ggplot(storage_country_data, aes(x = Country, y = Count, fill = Storage)) +
  geom_bar(stat = "identity", position = "fill") +
  labs(title = "各国家存储容量偏好 (百分比)",
       x = "国家",
       y = "比例",
       fill = "存储容量") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_y_continuous(labels = percent_format())

print(p9)

# 8.10 数量 vs 价格 散点图
p10 <- ggplot(data, aes(x = Price, y = Quantity, color = Country, size = Quantity)) +
  geom_point(alpha = 0.6) +
  labs(title = "价格与购买数量关系",
       x = "价格 ($)",
       y = "购买数量",
       color = "国家") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_smooth(method = "lm", se = FALSE, color = "gray", linetype = "dashed")

print(p10)

# 8.11 月度各型号销售趋势 - 多线图
monthly_model_data <- data %>%
  group_by(Month, iPhone_Model) %>%
  summarise(Total_Sales = sum(Quantity), .groups = "drop")

p11 <- ggplot(monthly_model_data, aes(x = Month, y = Total_Sales, color = iPhone_Model, group = iPhone_Model)) +
  geom_line(size = 1.2) +
  geom_point(size = 2.5) +
  labs(title = "各型号月度销售趋势",
       x = "月份",
       y = "销量",
       color = "型号") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5))

print(p11)

# 8.12 关联规则可视化
# 规则散点图
plot(rules, method = "scatterplot", 
     control = list(jitter = 2),
     shading = "lift",
     main = "关联规则散点图")

# 规则网络图 (前10条规则)
plot(head(rules_lift, 10), method = "graph",
     control = list(type = "items"),
     main = "关联规则网络图")

# 8.13 聚类结果可视化
# 客户聚类 - 散点图矩阵
pairs(cluster_features, col = kmeans_result$cluster, pch = 19,
      main = "客户聚类散点图矩阵")

# 聚类轮廓图
sil <- silhouette(kmeans_result$cluster, dist(cluster_scaled))
plot(sil, main = "聚类轮廓图", col = 1:k)

# 8.14 组合图形 - 销售仪表板
grid.arrange(p1, p2, p5, p6, ncol = 2, 
             top = "iPhone销售分析仪表板")

# 8.15 保存所有图表
ggsave("country_revenue.png", p1, width = 8, height = 6, dpi = 300)
ggsave("model_distribution.png", p2, width = 8, height = 6, dpi = 300)
ggsave("color_preference.png", p3, width = 8, height = 6, dpi = 300)
ggsave("payment_distribution.png", p4, width = 8, height = 6, dpi = 300)
ggsave("monthly_trend.png", p5, width = 8, height = 6, dpi = 300)
ggsave("heatmap_country_model.png", p6, width = 10, height = 6, dpi = 300)
ggsave("price_boxplot.png", p7, width = 8, height = 6, dpi = 300)
ggsave("price_histogram.png", p8, width = 8, height = 6, dpi = 300)
ggsave("storage_preference.png", p9, width = 8, height = 6, dpi = 300)
ggsave("price_quantity_scatter.png", p10, width = 8, height = 6, dpi = 300)
ggsave("monthly_model_trend.png", p11, width = 10, height = 6, dpi = 300)

cat("\n\n========== 分析完成 ==========\n")
cat("所有图表已保存到当前工作目录\n")