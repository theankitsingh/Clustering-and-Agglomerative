setwd("C:\\Users\\ankit\\python")

data_reduction = read.table("C:\\Users\\ankit\\python\\ect_data.txt", header = T, sep = "\t")
names(data_reduction)

data_reduction.pca = data_reduction[c("attitude1_01","attitude1_02","attitude1_03","attitude1_04","intent1_01","intent1_02","intent1_03","intent1_04","peruse01","peruse02","peruse03","peruse04","satis01","satis02","satis03","satis04")]
pcamodel_reduc = princomp(data_reduction.pca, cor = FALSE)
pcamodel_reduc$sdev^2

plot(pcamodel_reduc, main = "Screeplot for ect_data columns")

data_reduction.FA = factanal(~attitude1_01+attitude1_02+attitude1_03+attitude1_04+intent1_01+intent1_02+intent1_03+intent1_04+peruse01+peruse02+peruse03+peruse04+satis01+satis02+satis03+satis04, factors = 4, rotation = "varimax", scores = "none", data = data_reduction)
data_reduction.FA

install.packages("tree")

kmeans_data = read.table("C:\\Users\\ankit\\python\\car.test.frame.txt", header = T, sep = "\t")
par(mfrow = c(1,2))
kmd = kmeans(data.frame(kmeans_data$Mileage, kmeans_data$Price), 5)
plot(kmeans_data$Mileage, kmeans_data$Price, col=kmd[[1]], main = "5 KM Groups")
kmd1 = kmeans(data.frame(kmeans_data$Mileage, kmeans_data$Price), 8)
plot(kmeans_data$Mileage, kmeans_data$Price, col=kmd1[[1]], main = "8 KM Groups")

lcd_dist = dist(kmeans_data[,c(1,4,8)])
cd_clust = hclust(cd_dist)
plot(cd_clust,xlab="Mileage",ylab="Price",main= "Agglomerative clustering")