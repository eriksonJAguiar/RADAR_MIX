import torch
import torchvision.models as models
#from medmnist_models import ResNet50, ResNet18
import torch.nn as nn
import torch.nn.functional as F


class ModelsPretrained():
    
    def _initialize_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    
    def _freeze_layers(self, model, num_layers_to_freeze):
        '''
            function to freeze layer of CNNs
            params:
                - model: target model to freeze layers
                - num_layers_to_freeze: number of layer aim to freeze
        '''
        #freeze all layer
        for param in model.parameters():
            param.requires_grad = False
        
        #unfreeze the last layer defined by num_layers_to_freeze
        for child in list(model.children())[-num_layers_to_freeze:]:
            for param in child.parameters():
                param.requires_grad = True
     
    def make_model_pretrained(self, model_name, num_class):
        """function to select models pre-trained on image net and using tochvision architectures

        Args:
            model_name (str): string to describe the name of the models, such as Resnet50, Resnet18, inceptionv3, densenet201, vgg16, vgg18, and efficientnet
            num_class (int): number of class extracted from dataset

        Returns:
            pytorch.model: a loaded model using the pytorch structure
        """
        model = None
        out_features_model = num_class if num_class > 2 else 1

        if model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
            #self._freeze_layers(model, 5)
            for param in model.parameters():
                param.requires_grad = False
            
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(model.fc.in_features, 224),
                torch.nn.BatchNorm1d(224),
                torch.nn.ReLU(),
                #torch.nn.Dropout(0.5),
                #torch.nn.Linear(512, 128),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(224, out_features_model)
            )
            
            #model.apply(self._initialize_weights)
        
        elif model_name == "resnet50-he":
            model = models.resnet50()
            
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(model.fc.in_features, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                #torch.nn.Dropout(0.5),
                #torch.nn.Linear(512, 128),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model)
            )
            
            for m in model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
        
        elif model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
            model.fc = torch.nn.Sequential(
               torch.nn.Linear(model.fc.in_features, out_features_model),
               torch.nn.Softmax()
            )
            #model = ResNet18(3, out_features_model)
            #model = ResNet50(1, num_classes=num_class)
                
        elif model_name == "densenet":

            model = models.densenet.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            
            for param in model.parameters():
                param.requires_grad = False
            
            #self._freeze_layers(model, 10)
            
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model),
                #torch.nn.Linear(num_ftrs, out_features_model),
                #torch.nn.Softmax()
            )
            #model.classifier = nn.Linear(num_ftrs, out_features=out_features_model) 
        
        elif model_name == "inceptionv3":
            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
            model.aux_logits = False
            #model = models.inception_v3(pretrained=True, aux_logits=False)
            
            #self._freeze_layers(model, 10)
            
            for param in model.parameters():
                param.requires_grad = False

            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model),
            )

        elif model_name == "vgg16":
            model = models.vgg.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            
            for param in model.parameters():
                param.requires_grad = False
            
            #self._freeze_layers(model, 10)
            
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model),
            )
            #model.classifier[6].register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
            #model.classifier[6] = torch.nn.Linear(num_ftrs, out_features=out_features_model)
        
        
        elif model_name == "vgg19":
            model = models.vgg.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            
            for param in model.parameters():
                param.requires_grad = False
            
            #self._freeze_layers(model, 10)
            
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Sequential(
                #torch.nn.Linear(num_ftrs, out_features_model),
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model),
                # torch.nn.Dropout(0.5),
                # torch.nn.Linear(224, out_features_model),
                #torch.nn.Softmax(),
            )

        elif model_name == "efficientnet":
            model = models.efficientnet.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            
            for param in model.parameters():
                param.requires_grad = False
            
            #self._freeze_layers(model, 10)
            
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model),
                #torch.nn.Softmax(),
            )
            #model.classifier[1] = nn.Linear(num_ftrs, out_features=out_features_model)
        
        else:
            print("Ivalid model name, exiting...")
            exit()

        return model
        