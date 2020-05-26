import os
from removebg import RemoveBg
class ChangePic():
    def __init__(self,key,pic):
        self.key = key
        self.pic = pic

    def remove(self):
        """
        单张图片抠图
        :return:
        """
        rmove = RemoveBg(self.key,'error.log')
        path = self.pic
        print(path)
        rmove.remove_background_from_img_file(path)
if __name__ == '__main__':
    ChangePic('VNzVdb7uBAgryt7xkUcDDRXT','D:\\aaaa.jpg').remove()