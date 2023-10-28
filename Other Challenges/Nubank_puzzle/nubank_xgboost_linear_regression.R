install.packages("xgboost")
install.packages("stats")
install.packages("base")
install.packages("utils")
install.packages("graphics")

library("xgboost")
library("stats")
library("base")
library("utils")
library("graphics")


##ler o dataset
dataset<- read.csv("C:\\5dts\\train.csv",sep=",",dec=".")

head(dataset)
#dropar a coluna id
dataset <- subset(dataset, select = -id)


colunas_texto <- sapply(dataset, is.character)
colunas_texto <- colnames(dataset)[colunas_texto]


#aplicar label encoding nas colunas de texto
for (coluna in colunas_texto) {
  dataset[[coluna]] <- as.numeric(factor(dataset[[coluna]]))
}

head(dataset)
###Visualizar tipo de colunas

str(dataset)
##transformar todas as colunas para tipo numero
dataset <- as.data.frame(sapply(dataset, as.numeric))


# Criação de modelo teste para seleção de features
modelo_atual <- lm(target ~., data = dataset) 

#Seleção de features de regressão usando a tecnica de stepwise forward ou backward com comando step()
new_model <- step(modelo_atual)

summary(new_model)

#resultado de 58 features selecionadas (Redução de 45,79% do dataset)
features_selecionadas <- c(
  "X016399044a", "X023c68873b", "X04e7268385",
  "X06888ceac9", "X087235d61e", "X136c1727c3", 
  "X174825d438", "X1f3058af83", "X1fa099bb01", 
  "X253eb5ef11", "X25bbf0e7e7", "X2719b72c0d", 
  "X298ed82b22", "X29bbd86997", "X2a457d15d9", 
  "X2bc6ab42f7", "X2d7fe4693a", "X435dec85e2", 
  "X49756d8e0f", "X55907cc1de", "X56371466d7", 
  "X5b862c0a8f", "X5f360995ef", "X60ec1426ce", 
  "X63bcf89b1d", "X65aed7dc1f", "X6db53d265a", 
  "X789b5244a9", "X7fe6cb4c98", "X87b982928b", 
  "X8c2e088a3d", "X8f5f7c556a", "X98475257f7", 
  "X9a575e82a4", "X9b6e0b36c2", "a14fd026ce",  
  "abca7a848f", "ae08d2297e", "aee1e4fc85", 
  "bdf934caa7", "beb6e17af1", "c0c3df65b1", 
  "c58f611921", "d2c775fa99", "d4d6566f9c", 
  "dcfcbc2ea1", "e0a0772df0", "e16e640635", 
  "e7ee22effb", "e86a2190c1", "ea0f4a32e3", 
  "ed7e658a27", "f013b60e50", "f0a0febd35", 
  "f66b98dd69", "fbf66c8021", "fe0318e273", 
  "ffd1cdcfc1","target"
)


###remover outlier da regressao
boxplot(dataset$target, 
        main = "Boxplot da Variável Target", 
        ylab = "Valores da Target")  

limite_superior <- 2.88

###outliers removidos
dataset_filtrado <- subset(dataset, target <= limite_superior)


boxplot(dataset_filtrado$target, 
        main = "Boxplot da Variável Target", 
        ylab = "Valores da Target")  

###dataset selecionado com as melhores features + target
dataset_filtrado <- dataset[, features_selecionadas]


###criação de bases de reino e teste####
set.seed(123)  
observacoes <- nrow(dataset_filtrado)

t_treinamento <- 0.8
t_teste <- 0.2

t_treinamento <- round(observacoes * t_treinamento)
t_teste <- observacoes - t_treinamento
amostra <- sample(1:observacoes, observacoes)

# Criação de bases de treino e teste
base_treino <- dataset_filtrado[amostra[1:t_treinamento], ]
base_teste <- dataset_filtrado[amostra[(t_treinamento + 1):observacoes], ]


####APLICACAO INICIAL DO MODELO DE REGRESSAO LINEAR ANTES DO XGBOOST#####
#####RMSE=2.088745
####R2 =0.2704335

###########################################################################
modelo_atual <- lm(target ~ ., data = base_treino)
resumo<-summary(modelo_atual)
print(resumo$r.squared)

predict_atual <- predict(modelo_atual, newdata = base_teste)
erros_quadrados_rl_atual <- (base_teste$target - predict_atual)^2

rmse <- sqrt(mean(erros_quadrados_rl_atual))

print(rmse)
#############Realizando bosting com xgbost na regressao linear################

##criando uma matrix a partir dos dados de treinamento
xgtrain <- xgb.DMatrix(data = as.matrix(base_treino[, -1]), label = base_treino$target)
# criando uma matriz  a partir dos dados de teste
xgtest <- xgb.DMatrix(data = as.matrix(base_teste[, -1]), label = base_teste$target)

##parametrização do modelo
wlist <- list(val=xgtest, train=xgtrain)


avaliacaoerror <- function(predit, train) {
  labels <- getinfo(train, "label")
  err <- cor(predit, as.numeric(labels)) ^ 2
  return(list(metric = "error", value = err))}


parametros <- list(objective = "reg:linear",
              eta = 0.01,
              min_child_weight = 3,
              subsample = .8,
              colsample_bytree = .8,
              scale_pos_weight = 1.0,
              gamma = 0,
              subsample = 0.5,
              colsample_bytree = 0.5,
              max_depth = 8,
              eval_metric=avaliacaoerror)
#############criação do modelo xgboost de regressao linear usando a matriz de treino#############
####modelo irá ficar em loop "auto improve", irá dar break quando o proximo valor for ele mesmo,
####ou seja, chegará em um ponto que não há melhoria a ser feita
modelxg <- xgb.train(params = parametros, 
                 data = xgtrain, 
                 nround = 5000, 
                 watchlist=wlist, 
                 max_delta_step = 1,
                 maximize = TRUE)

#####predição do modelo####
values_predict <- predict(modelxg, xgtest)

########AVALIANDO O MODELO XGBOOST DE FORMA INICIAL COM R2 e RMSE#############
###RMSE=0.6277049
###R2=0.9314047

####OS INDICADORES CLARAMENTE MELHORARAM CONFORME MOSTRADO NA REGRESSAO ANTES DO XGBOOST
##################
# Calcular o erro quadrado
erros_quadradicos <- (base_teste$target - values_predict)^2

# Calcular o RMSE
rmse <- sqrt(mean(erros_quadradicos))

# Exibir o valor do RMSE
print(rmse) #####0.6277


# Calcular o R2
r2 <- 1 - sum((base_teste$target - values_predict)^2) / sum((base_teste$target - mean(base_teste$target))^2)

# Exibir o valor do R2
print(r2)#### 0.9314047




######VALIDANDO O MODELO, TESTE DE RESIDUOS#####
residuos <- base_teste$target - values_predict

residuos_dataf <- data.frame(values_predict = values_predict, residuals = residuos)


plot(residuals ~ values_predict, data = residuos_dataf,
            xlab = "Predict", ylab = "Resíduos",
            main = "Resíduos vs. Valores Previstos")
qqline(residuos)


###surpreendentemente parece resolver muito bem o problema, consegue predizer muito bem os valores