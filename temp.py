from moviepy.editor import *
import os


# 定义一个数组


def main(L, outMvName):
    # mvTemp = getFileList(inpath)
    final_clip = concatenate_videoclips(L)
    # 生成目标视频文件
    final_clip.to_videofile("./{}.mp4".format(outMvName), fps=24, remove_temp=False, )

path = 'F:\\temp'
name = 'combine'
main(path, name)