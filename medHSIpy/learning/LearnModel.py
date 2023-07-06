import torch

def train(model, train_dataloader, optimizer, criterion, device):
    model.train().to(device)
    running_loss = 0
    correct = 0
    total = 0
    
    for  batch in train_dataloader:
        batch_image = batch['image'].to(torch.float).to(device)
        batch_target = batch['label'].to(torch.long).to(device)
        
        """
        image_device = batch_image.device
        print(f'image device: {image_device}')
        target_device = batch_target.device
        print(f'target device: {target_device}')
        model_device = next(model.parameters()).device
        print(f'model device: {model_device}')
        """
        
        optimizer.zero_grad()
        
        outputs = model(batch_image)
        
        '''
        del batch_image
        torch.cuda.empty_cache()
        '''
        
        loss = criterion(outputs, batch_target)
        
        
        loss.backward()
        optimizer.step()
        
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == batch_target).sum().item()
        total += batch_target.size(0)
        
        '''
        del batch_target
        torch.cuda.empty_cache()
        '''
        
        
        '''
        del loss
        torch.cuda.empty_cache()
        '''
        
        
    train_loss = running_loss / len(train_dataloader) 
    train_acc = correct / total
    
    return train_loss, train_acc

def valid(model, valid_dataloader, criterion, device, is_test=False, result_file_path=None):
    model.eval().to(device)
    running_loss = 0
    correct = 0
    total = 0
    Correct_Data = []
    False_Data = []
    with torch.no_grad():
        for batch in valid_dataloader:
            batch_image = batch['image'].to(torch.float).to(device)
            batch_target = batch['label'].to(torch.long).to(device)
            
            """
            image_device = batch_image.device
            print(f'image device: {image_device}')
            target_device = batch_target.device
            print(f'target device: {target_device}')
            model_device = next(model.parameters()).device
            print(f'model device: {model_device}')
            """
            
            outputs = model(batch_image)
            
            loss = criterion(outputs, batch_target)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == batch_target).sum().item()
            total += batch_target.size(0)
            
            if is_test:
                for i, bool_ in enumerate(predicted == batch_target):
                    if bool_:
                        Correct_Data.append(batch['original_image'][i].numpy())
                    else:
                        False_Data.append(batch['original_image'][i].numpy())
                
                f = open(result_file_path, 'a')
                f.write(f'correct: {predicted.tolist()==batch_target.tolist()}, target: {batch_target.tolist()}, pred: {predicted.tolist()} \n')
                f.write(f'outputs: {outputs.tolist()} \n')
                f.write(f'loss: {loss}, running_loss: {running_loss} \n')
                f.write(f'{total} is end \n')
                f.close()
                
                print(f'correct: {predicted.tolist()==batch_target.tolist()}, target: {batch_target.tolist()}, pred: {predicted.tolist()}')
                print(f'outputs: {outputs.tolist()}')
                print(f'loss: {loss}, running_loss: {running_loss}')
                print(f'{total} is end')
            
        val_loss = running_loss / len(valid_dataloader)
        val_acc = correct / total
        
        if is_test:
            print(f'{correct} / {total} is accurate')
        
        return val_loss, val_acc, Correct_Data, False_Data

def Learn_Model(train_dataloader, valid_dataloader, model, num_of_epoch, criterion, optimizer, device, is_test=0):
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    
    if is_test == 0:
        disp_epoch = int(num_of_epoch // 10)
    else:
        disp_epoch = 1
    
    if disp_epoch == 0:
        disp_epoch = 1
    
    '''学習'''
    for epoch in range(num_of_epoch):        
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = valid(model, valid_dataloader, criterion, device, is_test=False)
        
        if (epoch + 1) % disp_epoch == 0:
            print('epoch %d, train loss: %.4f , train_acc: %.4f, val_loss: %.4f val_acc: %.4f' % (epoch, train_loss, train_acc, val_loss, val_acc))
            
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss_list.append(val_loss)
        valid_acc_list.append(val_acc)
        
    result = {
        'train':
        {
            'loss': train_loss_list, 
            'acc': train_acc_list}, 
        'test':
        {
            'loss': valid_loss_list, 
            'acc': valid_acc_list}
        }
    return model, result
