import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import os
from helpers.dataframe_utils import save_config
from sklearn.metrics import roc_curve, roc_auc_score

class LMCustomInterface(pl.LightningModule):
    def __init__(self, cols=['loss', 'ae_loss', 'cls_loss', 'tskin_loss', 'hr_loss', 
                'rmssd_loss', 'rr_loss', 'met_loss', 'acc', 'prec', 'rec'],
                 calc_auc_at_epoch_end=False,
                 calc_score_at_epoch_end=False
                ):
        super().__init__()
        self.calc_auc_at_epoch_end = calc_auc_at_epoch_end
        self.calc_score_at_epoch_end = calc_score_at_epoch_end
        
        if self.calc_auc_at_epoch_end:
            cols.append("auc")
        if self.calc_score_at_epoch_end:
            cols.append("score")
        
        self.tr_losses = []
        self.val_losses = []
#         self.register_buffer('tr_losses', [])
#         self.register_buffer('val_losses', [])
        
        self.cols = cols
#         self.register_buffer('cols', cols)
        
        self.df_tr_losses = pd.DataFrame([], columns=self.cols)
        self.df_val_losses = pd.DataFrame([], columns=self.cols)
#         self.register_buffer('df_tr_losses', pd.DataFrame([], index=self.cols))
#         self.register_buffer('df_val_losses', pd.DataFrame([], index=self.cols))
        
        self.vnum = self.get_vnum()
#         self.register_buffer('vnum', self.get_vnum())
        self.tr_fpath = self.get_savefname(dset='train')
#         self.register_buffer('tr_fpath', self.get_savefname(dset='train'))
        self.val_fpath = self.get_savefname(dset='valid')
#         self.register_buffer('val_fpath', self.get_savefname(dset='valid'))
        
        self.conf = {'version_num':self.vnum}
#         self.register_buffer('conf', {'version_num':self.vnum})
    
    def training_step(self, batch, batch_idx):
        """
        return logs: dictionary of String keys and torch.tensor loss/metric values
        
        e.g.
        x, y = batch
        x_hat, y_hat = self(x)
        tr_loss, ae_loss, cls_loss = self.compute_network_loss(x_hat, x, y_hat, y) 
        logs = {"loss": tr_loss, "ae_loss": ae_loss, "cls_loss": cls_loss}
        """
        print("Not yet implemented: training_step")
        assert(False)
        return logs
    
    def configure_optimizers(self):
        """optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=1e-3)
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         [60, 90], 
                                                         gamma=0.5)
        return [optimizer], [lr_sched]"""
        print("Not yet implemented: configure_optimizers()")
        assert(False)
        return None
    
    def init_model(self):
        print("Not yet implemented: init_model()")
        assert(False)
    
    def forward(self, x):
        print("Not yet implemented: forward()")
        assert(False)
        return None
    
    def add_to_config(self):
        print("Not yet implemented: config()")
        assert(False)
    
    def training_step_end(self, batch_parts):
        return batch_parts

    def training_epoch_end(self, training_step_outputs):
        losses, rstr = self.losses_from_outputs(training_step_outputs)
        self.tr_losses.append(losses)
        
        all_losses_dict = self.get_all_losses(training_step_outputs)
        
        if self.calc_auc_at_epoch_end:
            if len(np.unique(self.auc_labels_tr)) > 1:
                all_losses_dict['auc'] = roc_auc_score(self.auc_labels_tr, self.auc_logits_tr)
            else:
                all_losses_dict['auc'] = np.nan
            self.auc_labels_tr, self.auc_logits_tr = [], []
            rstr += f", auc: {all_losses_dict['auc']}"

        if self.calc_score_at_epoch_end:
            if len(np.unique(self.score_labels_tr)) > 1:
                all_losses_dict['score'] =  average_precision_score(self.auc_labels_tr, self.auc_logits_tr)
            else:
                all_losses_dict['auc'] = np.nan
            self.auc_labels_tr, self.auc_logits_tr = [], []
            rstr += f", auc: {all_losses_dict['auc']}"
        
        row = pd.DataFrame(all_losses_dict, 
                           index=[self.current_epoch])
        self.df_tr_losses = pd.concat([self.df_tr_losses, row], ignore_index=True)
        self.df_tr_losses.to_pickle(self.tr_fpath)
        print(f"Epoch {self.current_epoch}")
        print(f"\tTrain {rstr}")
    
    def validation_step_end(self, batch_parts):
        return batch_parts

    def validation_epoch_end(self, validation_step_outputs):
        losses, rstr = self.losses_from_outputs(validation_step_outputs)
        self.val_losses.append(losses)
        
        all_losses_dict = self.get_all_losses(validation_step_outputs)
        
        if self.calc_auc_at_epoch_end:
            if len(np.unique(self.auc_labels_val)) > 1:
                all_losses_dict['auc'] = roc_auc_score(self.auc_labels_val, self.auc_logits_val) 
            else:
                all_losses_dict['auc'] = np.nan
            self.auc_labels_val, self.auc_logits_val = [], []
            rstr += f", auc: {all_losses_dict['auc']}"
        
        row = pd.DataFrame(all_losses_dict, 
                           index=[self.current_epoch])
        self.df_val_losses = pd.concat([self.df_val_losses, row], ignore_index=True) 
        self.df_val_losses.to_pickle(self.val_fpath)
        print(f"\tValid {rstr}\n")
        
    def get_all_losses(self, step_outputs):
        epoch_losses = {}
        for c in self.cols:
            if c == 'auc':
                continue
            if c in ['prec', 'rec', 'f1', 'sup', 'acc', 'TPR', 'FNR', 'TP/TP+FP', 'FN/FN+TN']:
                c_loss = np.nanmean(np.hstack([output[c] for output in step_outputs]))
            else:
                c_loss = self.torch_nanmean(torch.stack([output[c] for output in step_outputs])).item()
            
            # c_loss = np.nanmean(np.hstack([output[c] for output in step_outputs]))
            epoch_losses[c] = c_loss
        return epoch_losses
        
    def torch_nanmean(self, x):
        num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum()
        value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum()
        return value / num
        
    def losses_from_outputs(self, step_outputs):
        def sort_keys(a):
            if a == 'loss':
                return 0
            elif 'loss' in a:
                if any([ kw in a for kw in ['tr', 'train']]):
                    return 1
                elif any([ kw in a for kw in ['val', 'valid']]):
                    return 2
                else:
                    return 3
            else: 
                return 4
            
        vals_for_epoch = {k:[] for k in sorted(step_outputs[0].keys(), 
                                               key=lambda a: sort_keys(a))}
        
        for out in step_outputs:
            for k, v in out.items():
                if k in ['prec', 'rec', 'auc', 'acc']:
                    v = v
                if type(v) in [torch.tensor, torch.Tensor]: 
                    v = v.item()
                vals_for_epoch[k].append(v)
        
        rstr = ""
        for k, v in vals_for_epoch.items():
            
            if type(v[0]) == np.ndarray:
                v = np.array([list(v_)[0] for v_ in v])
            else:
                v = np.array(v)
            rstr += f" {k}: {np.nanmean(v):.3f}" 
        return vals_for_epoch["loss"], rstr
    
    def get_vnum(self):
        return 1+max([int(dirname.split('_')[1]) for dirname in os.listdir('lightning_logs/') if dirname[0]=='v'])
        
    def get_savefname(self, dset='train'):
        return f"epoch_logs/{dset}_logs_{self.vnum}.pandas"
    

def softmax_np(x, axis=1):
    return np.exp(x)/np.sum(np.exp(x), axis=axis).reshape(-1,1)



class GigaSpaceNeedle(LMCustomInterface):
    def __init__(self,  
                 feats=config_ds['feat_cols'], 
                 learning_rate=1e-3,
                 lr_steps=[],
                 layer_defs=None,
                 use_cb_loss=False,
                 l2_reg=0,
                 loss_args={},
                 pos_class=0,
                 verbose=True
                ):
        super().__init__(cols=['loss', 'cls_loss', 'acc', 'prec_0', 'rec_0', 'prec_1', 'rec_1'], 
                         calc_auc_at_epoch_end=True, calc_score_at_epoch_end=True)
        self.feats = feats 
        self.N_FEATS = len(feats)
        self.lr = learning_rate
        self.lr_steps = lr_steps
        self.layer_defs = layer_defs
        self.use_cb_loss = use_cb_loss
        self.l2_reg = l2_reg
        self.loss_args = loss_args
        self.pos_class = pos_class
        self.verbose = verbose
        self.init_model()
        
        self.tmf_val_acc = torchmetrics.Accuracy(num_classes=2)#.to(self.device)
        self.tmf_tr_acc = torchmetrics.Accuracy(num_classes=2)#.to(self.device)
        self.tmf_val_prec = torchmetrics.Precision(num_classes=2, average='none')#.to(self.device)
        self.tmf_tr_prec = torchmetrics.Precision(num_classes=2, average='none')#.to(self.device)
        self.tmf_val_rec = torchmetrics.Recall(num_classes=2, average='none')#.to(self.device)
        self.tmf_tr_rec = torchmetrics.Recall(num_classes=2, average='none')#.to(self.device)

        # auc curve
        self.auc_logits_tr = []
        self.auc_labels_tr = []
        self.auc_logits_val = []
        self.auc_labels_val = []
        # prec-rec curve
        self.score_logits_tr = []
        self.score_labels_tr = []
        self.score_logits_val = []
        self.score_labels_val = []
        
        #self.tmf_val_acc = accuracy_np
        #self.tmf_tr_acc = accuracy_np
        
        
        self.add_to_config()
        print(self.conf)
        save_config(self.conf)
        print(self.get_savefname())
        if self.verbose: 
            print(self)
        
    def add_to_config(self):
        self.conf['network'] = 'GigaSpaceNeedle'
        self.conf['layer_defs'] = self.layer_defs    
        self.conf['feats'] = self.feats
        self.conf['use_cb_loss'] = self.use_cb_loss
        self.conf['l2_reg'] = self.l2_reg
        self.conf['lr_steps'] = self.lr_steps
        self.conf['lr'] = self.lr
        self.conf['loss_args'] = self.loss_args
        self.conf['pos_class'] = self.pos_class
        
    def training_step(self, batch, batch_idx):
        #print('1', self.device, torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        x, y, _ = batch
        #print('2',self.device, torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        #x = x.to(self.device, dtype=torch.float)
        y_hat = self(x)
        #print('3',self.device, torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        loss_dict = self.compute_network_loss(y_hat, y)
        #print('4',self.device, torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
    
        y = y.int()
        
        #print('5',self.device, torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        logs = {"loss": loss_dict['net'], 
                "cls_loss": loss_dict['cls'],
                # "acc": self.tmf_tr_acc(y_hat, y).item(),
                # "prec_0": self.tmf_tr_prec(y_hat, y)[0].item(),
                # "rec_0": self.tmf_tr_rec(y_hat, y)[0].item(),
                # "prec_1": self.tmf_tr_prec(y_hat, y)[1].item(),
                # "rec_1": self.tmf_tr_rec(y_hat, y)[1].item()
                "preds": y_hat,
                "target": y
               }
        #print('6',self.device, torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        
        #display(y_hat.cpu().detach().numpy())
        #y_pred = softmax_np(y_hat.cpu().detach().numpy())[:,1]
        #display(y_pred)
        
#         ##JAKE
        self.auc_logits_tr.extend(softmax_np(y_hat.cpu().detach().numpy())[:,1])
        self.auc_labels_tr.extend(y.cpu().numpy())
        
        self.score_logits_tr.extend(softmax_np(y_hat.cpu().detach().numpy())[:,1])
        self.score_labels_tr.extend(y.cpu().numpy())
        
        
        out = 'training_step (pre del) mem %:', torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        #print(out)
        torch.cuda.empty_cache()        
        del x
        #print('7',self.device, torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        del y
        #print('8',self.device, torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        del y_hat
        #print('9',self.device, torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        del _
        #print('10',self.device, torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        #print(self.device)
        #print(out)
        
        return logs
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        #x = x.to(self.device, dtype=torch.float)
        y_hat = self(x)
        y_hat.detach()
        loss_dict = self.compute_network_loss(y_hat, y) 
        #loss_dict.detach()
        y = y.int()
                
        logs = {"loss": loss_dict['net'], 
                "cls_loss": loss_dict['cls'].detach(),
                "preds": y_hat,
                "target": y
               }
        
        self.auc_logits_val.extend(softmax_np(y_hat.cpu().detach().numpy())[:,1])
        self.auc_labels_val.extend(y.cpu().numpy())
        
        self.score_logits_val.extend(softmax_np(y_hat.cpu().detach().numpy())[:,1])
        self.score_labels_val.extend(y.cpu().numpy())
        
        out = 'validation_step (pre del) mem %:', torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        del x
        
        del y
        del y_hat
        del _
        #print(self.device)
        #print(out)
        torch.cuda.empty_cache()        
        #del batch
        
        return logs
    
    def training_step_end(self, outputs):
        
        total_loss = sum(outputs['loss']) / len(outputs['loss'])
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=BATCH_SIZE)
        acc = self.tmf_tr_acc(outputs['preds'], outputs['target'])
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True, batch_size=BATCH_SIZE)
        prec = self.tmf_val_prec(outputs['preds'], outputs['target'])
        rec = self.tmf_val_rec(outputs['preds'], outputs['target'])
        self.log('train_prec_0', prec[0], on_step=False, on_epoch=True, logger=True, batch_size=BATCH_SIZE)
        self.log('train_prec_1', prec[1], on_step=False, on_epoch=True, logger=True, batch_size=BATCH_SIZE)
        self.log('train_rec_0', rec[0], on_step=False, on_epoch=True, logger=True, batch_size=BATCH_SIZE)
        self.log('train_rec_1', rec[1], on_step=False, on_epoch=True, logger=True, batch_size=BATCH_SIZE)
        
        

#             rstr += f", auc: {all_losses_dict['auc']}"

#         if self.calc_score_at_epoch_end:
#             if len(np.unique(self.score_labels_tr)) > 1:
#                 all_losses_dict['score'] = average_precision_score(self.score_labels_tr, self.score_logits_tr)
#             else:
#                 all_losses_dict['score'] = np.nan
#             self.score_labels_tr, self.score_logits_tr = [], []
#             rstr += f", score: {all_losses_dict['score']}"
        
#         row = pd.DataFrame(all_losses_dict, 
#                            index=[self.current_epoch])
#         self.df_tr_losses = pd.concat([self.df_tr_losses, row], ignore_index=True)
#         self.df_tr_losses.to_pickle(self.tr_fpath)
        # print(f"Epoch {self.current_epoch}")
        #print(f"\tTrain {logs}")
        return total_loss
    
    def training_epoch_end(self, output):
        print(self.current_epoch)
        auc = 0
        if self.calc_auc_at_epoch_end:
            if len(np.unique(self.auc_labels_tr)) > 1:
                auc = roc_auc_score(self.auc_labels_tr, self.auc_logits_tr)
            else:
                auc = np.nan
            self.auc_labels_tr, self.auc_logits_tr = [], []
            self.log('train_auc', auc, on_step=False, on_epoch = True, logger=True, batch_size=BATCH_SIZE)
            
        prc = 0
        if self.calc_score_at_epoch_end:
            if len(np.unique(self.score_labels_tr)) > 1:
                prc = average_precision_score(self.score_labels_tr, self.score_logits_tr)
                
            else:
                prc = np.nan
                
            self.score_labels_tr, self.score_logits_tr = [], []
            self.log('train_prc', prc, on_step=False, on_epoch = True, logger=True, batch_size=BATCH_SIZE)
            
        return None
    
    def validation_step_end(self, outputs):

        total_loss = sum(outputs['loss']) / len(outputs['loss'])
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=BATCH_SIZE)
        acc = self.tmf_tr_acc(outputs['preds'], outputs['target'])
        self.log('val_acc', acc, on_step=False, on_epoch=True, logger=True, batch_size=BATCH_SIZE)
        prec = self.tmf_val_prec(outputs['preds'], outputs['target'])
        rec = self.tmf_val_rec(outputs['preds'], outputs['target'])
        self.log('val_prec_0', prec[0], on_step=False, on_epoch=True, logger=True, batch_size=BATCH_SIZE)
        self.log('val_prec_1', prec[1], on_step=False, on_epoch=True, logger=True, batch_size=BATCH_SIZE)
        self.log('val_rec_0', rec[0], on_step=False, on_epoch=True, logger=True, batch_size=BATCH_SIZE)
        self.log('val_rec_1', rec[1], on_step=False, on_epoch=True, logger=True, batch_size=BATCH_SIZE)
        
#         if self.calc_auc_at_epoch_end:
#             if len(np.unique(self.auc_labels_val)) > 1:
#                 all_losses_dict['auc'] = roc_auc_score(self.auc_labels_val, self.auc_logits_val) 
#                 self.fpr_val, self.tpr_val, _ = roc_curve(self.auc_labels_val, self.auc_logits_val)
#             else:
#                 all_losses_dict['auc'] = np.nan
#                 self.fpr_val = []
#                 self.tpr_val = []
            
#             self.auc_labels_val, self.auc_logits_val = [], []
#             rstr += f", auc: {all_losses_dict['auc']}"
            
#         if self.calc_score_at_epoch_end:
#             if len(np.unique(self.score_labels_val)) > 1:
#                 all_losses_dict['score'] = average_precision_score(self.score_labels_val, self.score_logits_val) 
#                 self.prec_val, self.rec_val, _ = precision_recall_curve(self.score_labels_val, self.score_logits_val, pos_label = 1)
#             else:
#                 all_losses_dict['score'] = np.nan
#                 self.prec_val = []
#                 self.rec_val = []
                
#             self.score_labels_val, self.score_logits_val = [], []
#             rstr += f", score: {all_losses_dict['score']}"
        
#         row = pd.DataFrame(all_losses_dict, 
#                            index=[self.current_epoch])
#         self.df_val_losses = pd.concat([self.df_val_losses, row], ignore_index=True) 
#         self.df_val_losses.to_pickle(self.val_fpath)
        #print(f"\tValid {logs}\n")
        return total_loss
        
    def validation_epoch_end(self, output):
        auc = 0
        if self.calc_auc_at_epoch_end:
            if len(np.unique(self.auc_labels_val)) > 1:
                auc = roc_auc_score(self.auc_labels_val, self.auc_logits_val) 
                self.fpr_val, self.tpr_val, _ = roc_curve(self.auc_labels_val, self.auc_logits_val)
            else:
                auc = np.nan
                self.fpr_val = []
                self.tpr_val = []
            
            self.auc_labels_val, self.auc_logits_val = [], []
            self.log('val_auc', auc, on_step=False, on_epoch = True, logger=True, batch_size=BATCH_SIZE)
        
        prc = 0
        if self.calc_score_at_epoch_end:
            if len(np.unique(self.score_labels_val)) > 1:
                prc = average_precision_score(self.score_labels_val, self.score_logits_val) 
                self.prec_val, self.rec_val, _ = precision_recall_curve(self.score_labels_val, self.score_logits_val, pos_label = 1)
            else:
                prc = np.nan
                self.prec_val = []
                self.rec_val = []
                
            self.score_labels_val, self.score_logits_val = [], []
            self.log('val_prc', prc, on_step=False, on_epoch = True, logger=True, batch_size=BATCH_SIZE)
            
        return None
        
    def compute_network_loss(self, cls_preds, cls_labels):
        #print("PR? - CNL")
        res = {}
        cls_loss = self.cross_entropy_classbalanced(cls_preds, cls_labels, 
                                                   beta=self.loss_args['cb_beta'])
        #cls_loss = F.cross_entropy(cls_preds, cls_labels)
        #print("NO - CNL")
        res['cls'] = cls_loss
        res['net'] = cls_loss
        return res
              
    def get_classbalance_weights(self, preds, labels, beta=0.999):
        # Get class counts
        classes, counts = torch.unique(labels, return_counts=True)
        if 1 not in classes: # Case where there are no sick samples
            counts = torch.tensor([counts[0], 0]).type_as(preds)#.device)
        if 0 not in classes: # Case where there are no healthy samples
            counts = torch.tensor([0, counts[0]]).type_as(preds)#.device)
        # Calculate weight for each class
        beta = torch.tensor(beta).type_as(preds)#.device)        
        one = torch.tensor(1.).type_as(preds)#.device)
        weights = (one - torch.pow(beta, counts)) / (one - beta)
        return weights        
    
    def cross_entropy_classbalanced(self, preds, labels, beta=0.9):
        weights = self.get_classbalance_weights(preds, labels, beta=beta)
        cb_ce_loss = F.cross_entropy(preds, labels, weight=weights)
        return cb_ce_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=self.l2_reg)
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer,self.lr_steps,gamma=0.1)
        return [optimizer], [lr_sched]
    
    def unwrap_layer_defs(self, layer_defs, drop_relu=False):
        layers = []
        for layer_def in layer_defs:
            if layer_def['type'] == 'linear':
                layers.append(nn.Linear(layer_def['n_in'], layer_def['n_out']))
                if not drop_relu:
                    layers.append(nn.ReLU())
            elif layer_def['type'] == 'conv1d':
                layers.append(nn.Conv1d(layer_def['n_in'], 
                                    layer_def['n_kernels'], 
                                    kernel_size=layer_def['kernel_size'], 
                                    stride=layer_def['stride'], 
                                    padding=layer_def['padding']))
                if not drop_relu:
                    layers.append(nn.ReLU())
            elif layer_def['type'] == 'conv2d':
                layers.append(nn.Conv2d(layer_def['n_in'],
                                        layer_def['n_kernels'],
                                        kernel_size=layer_def['kernel_size'],
                                        stride=layer_def['stride'],
                                        padding=layer_def['padding']))
                if not drop_relu:
                    layers.append(nn.ReLU())
            elif layer_def['type'] == 'lstm':
                layers.append(LSTMCustomLayer(num_layers=layer_def['num_layers'], 
                                              input_size=layer_def['input_size'], 
                                              hidden_size=layer_def['hidden_size'], 
                                              batch_size=layer_def['batch_size'],
                                              bidirectional=layer_def['bidirectional']))                
            elif layer_def['type'] == 'convtranspose1d':
                layers.append(nn.ConvTranspose1d(layer_def['dec_n_in'], 
                                    layer_def['dec_n_kernels'], 
                                    kernel_size=layer_def['dec_kernel_size'], 
                                    stride=layer_def['dec_stride'], 
                                    padding=layer_def['dec_padding']))
                if not drop_relu:
                    layers.append(nn.ReLU())
            elif layer_def['type'] == 'flatten': 
                layers.append(nn.Flatten())
            elif layer_def['type'] == 'unflatten':
                layers.append(nn.Unflatten(1, (layer_def['n_classes'], layer_def['n_datapoints'])))
            elif layer_def['type'] == 'maxpool1d':
                layers.append(nn.MaxPool1d(layer_def['kernel_size'],
                                           layer_def['stride'],
                                           layer_def['padding']))
            elif layer_def['type'] == 'maxpool2d':
                layers.append(nn.MaxPool2d(layer_def['kernel_size'],
                                           layer_def['stride'],
                                           layer_def['padding']))
            elif layer_def['type'] == 'dropout':
                layers.append(nn.Dropout(layer_def['p']))
            elif layer_def['type'] == 'batchnorm1d':
                layers.append(nn.BatchNorm1d(layer_def['n_feats']))
        return nn.Sequential(*layers)
    
    
    def init_model(self):
        # setting an encoder for each of features
        #encoder_list = []
        #for n in range(self.N_FEATS):    
        #    encoder_list.append(self.unwrap_layer_defs(self.layer_defs['encoder']))    
        #self.encoder_list = encoder_list
        self.encoder_list = nn.ModuleList([self.unwrap_layer_defs(self.layer_defs['encoder']) for n in range(self.N_FEATS)])
    
        self.cls_fc = self.unwrap_layer_defs(self.layer_defs['classifier'], drop_relu=True)
        #self.gradients = None
        
    def forward(self, x):
        # encoding each feature
        z = self.encode(x)
        del x
        assert(len(z) == self.N_FEATS)
        # concatenating the encodings
        z = torch.cat(z, 1)
        # classification on the encodings
        y_hat = self.cls_fc(z)
        del z
        return y_hat
    
    def encode(self, x):
        z = []
        # encoding each feature in the data 
        for i in range(self.N_FEATS):
            x_ = x[:,i,...]#.reshape(16,3,500,700)#.unsqueeze(1)
            z_ = self.encoder_list[i](x_)
            del x_
            z.append(z_)
            del z_
        del x
        return z
    
    def predict_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        del x
        del y
        y_pred = torch.argmax(y_hat,dim=1).int()
        return y_pred

def softmax_np(x, axis=1):
    return np.exp(x)/np.sum(np.exp(x), axis=axis).reshape(-1,1)
    
def accuracy_np(preds, target):
    if preds.shape[0] != target.shape[0]:
        return "bad input"
    preds_g = preds.argmax(dim=1)
    acc_sum = torch.sum(preds_g == target)
    return acc_sum / target.shape[0]