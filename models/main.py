import pandas as pd
import pandas as pd
from models import ModelTrainer

df=pd.read_csv(r"D:\course\Courses\NTI final project\NTI-final-project-\data\row\preprossing\final_EDA_.csv")
training=ModelTrainer(df)
result=training.train_and_evaluate()
for model,matrics in result.items():
    print(f'{model} Results :')
    for metric,score in matrics.items():
        print(f'{metric}: {score}')
