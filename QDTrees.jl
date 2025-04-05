using DataFrames, CSV, DecisionTree 

#function to display DataFrames in VSCode
display_df(df) = VSCodeServer.vscodedisplay(df)

# load data
BC_data = CSV.read("WisconsinBreastCancer.csv", DataFrame)

# define the features and target variable
y = BC_data[!, :diagnosis]
X = select(BC_data, Not(:diagnosis,:id,:Column33))
features = Matrix(X)

# Build a random forest using all 30 features, max depth of 4, and 50 trees

Forestmodel = build_forest(y,features, 30, 50, 0.5, 4)