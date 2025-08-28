#UI

ExpoRouter 将帮助我们在应用内部进行导航。

**Expo Router** 是基于 React Navigation 的一个路由框架，专为 Expo 和 React Native 应用设计，提供了一种文件系统驱动（file-based）的路由方案，使得构建和维护复杂导航结构变得更加简单和直观。

## **适用场景**
- 快速构建中小型 React Native 应用
- 需要基于文件结构管理页面和路由的项目
- 想要简化路由配置，减少导航代码的开发者
- Expo 应用开发者，期望享受无缝集成的路由解决方案

其遵循基于文件的导航系统，也就是说，只需要在App目录下创建文件，ExpoRouter就会为我们创建一个Screen

## 创建一个新的页面

在`src\app`下创建新的页面`index.jsx`

然后，根据[官方教程](https://docs.expo.dev/router/installation/#manual-installation)完成配置。一直做到Step 3:
1. 执行第一个npx命令
2. 修改`package.json`的`main`字段
3. 在`app.json`添加对应字段，value修改为项目名称。

教程后续配置可以继续扩展为web通用架构，暂时不加。

Step 4可以跳过，现在Expo可以自动为你处理这部分的修改。

然后执行Clear bundler cache

```
npx expo start --clear
```

因为此时已经修改了路由，不会再从根目录寻找页面，可能会报错（之前创建的index暂无内容）

![[indexEmptyError.png]]



