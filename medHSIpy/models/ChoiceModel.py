from . import Models2D, Models3D

def Choice_Model(model_name, image, target_class):
    if model_name == 'sample_CNN':
        model = Models2D.sample_CNN(num_of_class=len(target_class))
    elif model_name == 'original_CNN':
        model = Models2D.original_CNN(num_of_class=len(target_class))
    elif model_name == 'sample_3DCNN':
        model = Models3D.sample_3DCNN(image=image, num_of_class=len(target_class))
    elif model_name == 'vggLike_3DCNN':
        model = Models3D.vggLike_3DCNN(image=image, num_of_class=len(target_class))
    elif model_name == 'ResNetLike_3DCNN':
        model = Models3D.ResNetLike_3DCNN(image=image, num_of_class=len(target_class))
    elif model_name == 'my_3DCNN':
        model = Models3D.my_3DCNN(num_of_class=len(target_class))
    else:
        raise ValueError('Not found')
    return model

if __name__ == '__main__':
    target_class = [0, 1]
    model_name = 'sample_CNN'
    model = Choice_Model(model_name, target_class)
    print(model)