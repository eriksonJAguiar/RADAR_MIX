import torch
import pandas as pd
import os
from clean_train.train_model_lightning import TrainModelLigthning, CustomTimeCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
import lightning as L

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

device = torch.device(dev)
print("Device: {}".format(device))

#Class for training and test 
class PytorchTrainingAndTest():
    """Training and testing model
    """    
    def run_model(self, exp_num, model, model_name, database_name, train, test, learning_rate, num_epochs, metrics_save_path, num_class=2, is_per_class=False):
        """execute train and test of the model

        Args:
            exp_num (int): index of experiment executed
            model (torch.nn.Module): model will be trained
            model_name (str): model name
            database_name (str): database name
            train (torch.data.utils.Dataloader): train dataloader
            test (torch.data.utils.Dataloader): test datalaoder
            learning_rate (float): learning rate for training the model
            num_epochs (int): number of epochs for training and testing
            metrics_save_path (str): root path to save metrics
            num_class (int, optional): number of class in the dataset. Defaults to 2.
            is_per_class (bool, optional): If is True is calculated metrics per class. Defaults to False.

        Returns:
            metrics_df (pd.Dataframe): dataframe with model's performance
        """        
        #init model using a pytorch lightining call
        ligh_model = TrainModelLigthning(model_pretrained=model, 
                                         num_class=num_class, 
                                         lr=learning_rate,
                                         is_per_class=is_per_class)
        
        #define callback for earlystopping
        early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=8, verbose=True, mode='max')
        
        #define custom callback to calculate the train and test time 
        timer = CustomTimeCallback(os.path.join(metrics_save_path, "time", "train_time_{}-{}.csv".format(model_name, database_name)),
                                   os.path.join(metrics_save_path, "time", "test_time_{}-{}.csv".format(model_name, database_name)), 
                                   metrics_save_path)
            
        #Define callback to save the best model weights
        ckp = ModelCheckpoint(dirpath=os.path.join(metrics_save_path,"logs", "hold-out"), 
                              filename="{}-{}-exp{}".format(model_name, database_name, exp_num), 
                              save_top_k=1, 
                              mode="max", 
                              monitor="val_acc",
                              )
            
        #initate callbacks to execute the training
        callbacks=[early_stop_callback, ckp, timer]
        #callbacks=[ckp, timer]
        
        #define the function to save the logs
        logger = CSVLogger(save_dir=os.path.join(metrics_save_path,"logs", "hold-out"), name="{}-{}".format(model_name, database_name), version=exp_num)
        
        trainer = L.Trainer(
            max_epochs= num_epochs,
            accelerator="gpu",
            devices="auto",
            min_epochs=5,
            log_every_n_steps=10,
            logger=logger,
            deterministic=False,
            callbacks=callbacks
        )
            
        trainer.fit(
            model=ligh_model,
            train_dataloaders=train,
            val_dataloaders=test
        )
        
        metrics = trainer.logged_metrics
        print(metrics)
            
        if is_per_class:
            results =  {
                    "exp_num": exp_num,
                    "model_name" : model_name,
                    "train_loss":  metrics["loss"].item(),
                    "val_loss":  metrics["val_loss"].item(),
                }
            
            for i in range(num_class):
                results[f"train_acc_{i}"] =  metrics[f"acc_{i}"].item()
                results[f"train_precision_{i}"] =  metrics[f"precision_{i}"].item()
                results[f"train_recall_{i}"] =  metrics[f"recall_{i}"].item()
                results[f"train_spc_{i}"] =  metrics[f"specificity_{i}"].item()
                results[f"rain_f1-score_{i}"] =  metrics[f"f1_score_{i}"].item()
                
                results[f"val_acc_{i}"] =  metrics[f"val_acc_{i}"].item()
                results[f"val_precision_{i}"] =  metrics[f"val_precision_{i}"].item()
                results[f"val_recall_{i}"] =  metrics[f"val_recall_{i}"].item()
                results[f"val_spc_{i}"] =  metrics[f"val_specificity_{i}"].item()
                results[f"val_f1-score_{i}"] =  metrics[f"val_f1_score_{i}"].item()
        
        else:
            results =  {
                    "exp_num": exp_num,
                    "model_name" : model_name,
                    "train_acc" : metrics["acc"].item(),
                    "train_f1-score": metrics["f1_score"].item(),
                    "train_loss":  metrics["loss"].item(),
                    "train_precision":  metrics["precision"].item(),
                    "train_recall" :  metrics["recall"].item(),
                    "train_auc":  metrics["auc"].item(),
                    "train_spc":  metrics["specificity"].item(),
                    "val_acc" :  metrics["val_acc"].item(),
                    "val_f1-score":  metrics["val_f1_score"].item(),
                    "val_loss":  metrics["val_loss"].item(),
                    "val_precision":  metrics["val_precision"].item(),
                    "val_recall" :  metrics["val_recall"].item(),
                    "val_auc":  metrics["val_auc"].item(),
                    "val_spc":  metrics["val_specificity"].item(),
            }
        results = {k:[v] for k,v in results.items()}
        metrics_df = pd.DataFrame(results)
        
        metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

        self.save_metrics_to_figure(metrics, model_name, database_name)
        
        return metrics_df
    
    def run_model_kfold(self, exp_num, model, model_name, database_name, train, test, learning_rate, num_epochs, num_class=2, num_folds=5):
        """function to train the model using pytorch lightning

        Args:
            exp_num (int): num of experiments to run a model
            model (toch.nn.Module): architecture pre-trained that trained on dataset
            model_name (str): string that describe the model name
            database_name (str): string that describe the databaset to train a model
            train (Dataloader): dataloader of the train
            test (Dataloader): dataloader of the test
            learning_rate (float): learning rate of the model optimization
            num_epochs (int): number maximium of epoch during the trainning
            num_class (int, optional): number of class. Defaults to 2.
            num_folds (int, optional): number of folds on cross validation. Defaults to 5.

        Returns:
            metrics_final (pd.Dataframe): dataframe of metrics for training and testing
        """
        #init model using a pytorch lightining call
        ligh_model = TrainModelLigthning(model_pretrained=model, 
                                         num_class=num_class, 
                                         lr=learning_rate)
        
        #define callback for earlystopping
        early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=15, verbose=True, mode='max')
        
        #define custom callback to calculate the train and test time 
        timer = CustomTimeCallback("../metrics/time/train_time_{}-{}.csv".format(model_name, database_name),
                                       "../metrics/time/test_time_{}-{}.csv".format(model_name, database_name))
            
        metrics_final = pd.DataFrame()
        for i in range(num_folds):
          #Define callback to save the best model weights
          print("===== Starting fold {}/{} =====".format(i+1, num_folds))
          #define the function to save the logs
          logger = CSVLogger(save_dir=f"../metrics/logs/{model_name}_{database_name}/fold{i}", name="{}-{}".format(model_name, database_name), version=exp_num)
          ckp = ModelCheckpoint(dirpath=f"../metrics/logs/{model_name}_{database_name}/fold{i}", 
                                filename="{}-{}".format(model_name, database_name), 
                                save_top_k=1, 
                                mode="max", 
                                monitor="val_acc",
                                )
              
          #initate callbacks to execute the training
          callbacks=[early_stop_callback, ckp, timer]
          #callbacks=[ckp, timer]
          
          trainer = L.Trainer(
              max_epochs= num_epochs,
              accelerator="gpu",
              devices="auto",
              log_every_n_steps=10,
              logger=logger,
              deterministic=False,
              callbacks=callbacks
          )
              
          trainer.fit(
              model=ligh_model,
              train_dataloaders=train[i],
              val_dataloaders=test[i]
          )
          
          metrics = trainer.logged_metrics
          print(metrics)
              
          results =  {
                  "exp_num": exp_num,
                  "fold": i,
                  "model_name" : model_name,
                  "train_acc" : metrics["acc"].item(),
                  "train_f1-score": metrics["f1_score"].item(),
                  "train_loss":  metrics["loss"].item(),
                  "train_precision":  metrics["precision"].item(),
                  "train_recall" :  metrics["recall"].item(),
                  "train_auc":  metrics["auc"].item(),
                  "train_spc":  metrics["specificity"].item(),
                  "val_acc" :  metrics["val_acc"].item(),
                  "val_f1-score":  metrics["val_f1_score"].item(),
                  "val_loss":  metrics["val_loss"].item(),
                  "val_precision":  metrics["val_precision"].item(),
                  "val_recall" :  metrics["val_recall"].item(),
                  "val_auc":  metrics["val_auc"].item(),
                  "val_spc":  metrics["val_specificity"].item(),
          }
          results = {k:[v] for k,v in results.items()}
          metrics_df = pd.DataFrame(results)
          metrics_final = pd.concat([metrics_final, metrics_df])
          print(metrics_final)
        
          #metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

          #self.save_metrics_to_figure(metrics, model_name, database_name, fold=i)
        
        return metrics_final
      
    def save_metrics_to_figure(self, metrics, model_name, database_name, fold="hold-out"):
        '''
          Generate and save figures of metrics accuracy, loss, and f1-score by models and database
          
          Args:
            metris (dict): dict of metrics extract from pytoch.
            log_dir (str): path to save images.
            model_name (str): trained model name.
            database_name (str): database applied to train the model.
            fold (str, optional): If fold is not set up the hold-out is saved. Defaults hold-out.
        '''
        aggreg_metrics = []
        agg_col = "epoch"
        for i, dfg in metrics.groupby(agg_col):
            agg = dict(dfg.mean())
            agg[agg_col] = i
            aggreg_metrics.append(agg)

        #dataframe of metrics
        df_metrics = pd.DataFrame(aggreg_metrics)
        
        fold_name = fold if fold == "hold-out" else f"fold{fold}"
        
        os.makedirs(os.path.join("../", "metrics", "figures", fold_name), exist_ok=True)
        
        #save figures 
        fig_loss = df_metrics[["loss", "val_loss"]].plot(grid=True, legend=True, xlabel='Epoch', ylabel='Loss').get_figure()
        plt.tight_layout()
        fig_loss.savefig(os.path.join("../", "metrics", "figures", fold_name, "{}-{}-{}.png".format('loss_metrics',model_name, database_name)))
        
        fig_acc = df_metrics[["acc", "val_acc"]].plot(grid=True, legend=True, xlabel='Epoch', ylabel='Accuracy').get_figure()
        plt.tight_layout()
        fig_acc.savefig(os.path.join("../", "metrics", "figures", fold_name, "{}-{}-{}.png".format('acc_metrics',model_name, database_name)))
        
        fig_acc = df_metrics[["f1_score", "val_f1_score"]].plot(grid=True, legend=True, xlabel='Epoch', ylabel='F1-Score').get_figure()
        plt.tight_layout()
        fig_acc.savefig(os.path.join("../", "metrics", "figures", fold_name, "{}-{}-{}.png".format('f1_metrics',model_name, database_name)))
        
        fig_acc = df_metrics[["auc", "val_auc"]].plot(grid=True, legend=True, xlabel='Epoch', ylabel='AUC').get_figure()
        plt.tight_layout()
        fig_acc.savefig(os.path.join("../", "metrics", "figures", fold_name, "{}-{}-{}.png".format('auc_metrics',model_name, database_name)))
