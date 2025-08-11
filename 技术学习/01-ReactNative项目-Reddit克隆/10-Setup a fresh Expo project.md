## 创建项目

终端初始化项目
```cmd
npx create-expo-app@latest RedditClone --template
```

选择：
`Blank (TypeScrip)`


运行项目：
```
npm start
```

手机下载`Expo Go`后可以扫描二维码查看程序效果。

或者：

执行后输入`i`打开IOS模拟器（如果有）
输入`a`打开安卓模拟器
> 这需要提前保证你的电脑下载了Xcode或者Android Studio


尝试修改`App.tsx`中对应文字

![[project-init.png]]

## 为项目构建创建文件

为了保持项目的条理性，建议将所有源代码存储在一个名为`src`的文件夹中。  

1. 在项目根目录下创建一个名为“src”的新文件夹。  
2. 在‘ src ’中创建两个子文件夹：  
	1. **“app”**（用于屏幕和导航）  
	2. **“componets”**（用于可重用的UI组件）

为项目下载一些假数据：
https://notjust.notion.site/Reddit-Clone-Guide-19eb0ec93c5a801ca2a2d4feecf4d525?p=19fb0ec93c5a80a687bae8f56c194aac&pm=s


