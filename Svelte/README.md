# Svelte
1. SvelteKit, Svelte Native
2. svelte.dev

## Install 5.1.0
```
conda create --name Svelte
conda install 
pip install 

npm init vite my-svelte -- --template svelte
cd my-svelte
npm install
npm run dev
```

## Html
1. <div> 是一个块级元素，通常用于创建一个内容块，它会在前后形成换行。
2. <span> 是一个内联元素，适用于不需要换行的内容。

## JS
1. `null` 在 JavaScript 中表示“没有值”, `undefined` 中表示“值没定义”

## Tree
1. [Svelte Treeview](https://github.com/KeenMate/svelte-treeview)
2. [svelte-tree-viewer](https://github.com/kpulkit29/svelte-tree-viewer)
3. [svelte-tree-view-component](https://github.com/SamuelQZQ/svelte-tree-view-component)
4. [svelte-tree](https://github.com/mowtwo/svelte-tree)
5. [svelte-tree](https://github.com/esinx/svelte-tree)

## Svelte Material UI
1. Accordion/Nested Panel
2. Split button
3. select menu
4. Slider/"Range slider"
5. Text Field/File
6. Layout Grid
7. Radio List
8. Menu: Selection groups
9. Menu: Portal 多级的菜单
10. Menu 是一个专门用于显示选项列表的组件，适合传统的菜单场景
11. Menu Surface 提供了更广泛的用途，允许创建包含各种内容的弹出面板
12. Snackbar
13. Tabs
14. Tooltip

## Svelte Baisc
1. `slot` 被组件的使用者填充自定义的内容
2. 反应性是由赋值触发的, `num = num;`, `num=[...num, ]`
3. Spread 操作符 (...): 这个操作符用于将一个数组的所有元素展开, 代替`num=num.push(); num=num;`
4. 变量可以在html赋值 `<Nested answer={42} />`
5. 如果外面已经有`{}`，里面不能有`{}`
6. 有条件地渲染一些标记 `{#if c>10} <p/> {:else if c<5} <p/> {:else} <p/> {/if}`, `#`表示块开始。`:`表示连续, `/`表示结束。
7. 循环 `{#each colors as color, index} <p/> {/each}`, [颜色选择](https://learn.svelte.dev/tutorial/each-blocks)
8. 块的每次迭代指定一个`key`: 目标 **只想删除第一个<Thing>**组件及其 DOM 节点，而其他组件不受影响
9. 异步数据: `{#await promise} 	<p/>  {:then number} <p>{number}</p> {:catch error} <p>{error.message}</p> {/await}`
10. 不想在承诺解决之前显示任何内容: `{#await promise then number} <p>{number}</p> {/await}`, [异步数据](https://learn.svelte.dev/tutorial/await-blocks)
11. 循环可以包含内联事件, 编译器会优化
12. 事件处理可以具有修饰符 `once`, `preventDefault`阻止表单的默认提交行为-页面重新加载, `stopPropagation`, `passive`, `nonpassive`, `capture`, `self`是元素本身时才触发处理程序, `trusted`是用户操作，而不是 JavaScript 调用
13. 调度: `import { createEventDispatcher } from 'svelte';  const dispatch = createEventDispatcher();  dispatch('message', {text: 'Hello!'});`
14. 与 DOM 事件不同，组件事件不会冒泡传播。如果要侦听某个深层嵌套组件上的事件，中间组件必须转发该事件
15. 转发所有事件**简写** `<Inner on:message />`。否则默认只告知父亲
16. 名称和值一样**简写** `bind:value={value}`, `bind:value`
17. DOM 事件必须使用事件转发？[和14矛盾？？](https://learn.svelte.dev/tutorial/dom-event-forwarding)
18. 事件冒泡（Bubbling）: 从底到根`document`
19. 事件捕获（Capturing）: 从顶向下
20. 通常数据流是自上而下, <input>笨办法是使用`Event`, 最好`<input bind:value={name}>`
21. 在 DOM 中，一切都是字符串. `bind:value`自动转换类型
22. 复选框绑定到`bind:checked`, `let flavours = [];`
23. 表单提交-列表[<`select bind:value={selected}`](https://learn.svelte.dev/tutorial/select-bindings)
24. 一组 `bind:group={}`, 
25. 根据`value={number}`值选择`radio`[值选择/结果](https://learn.svelte.dev/tutorial/group-inputs)
26. 只能PC: 多选 `<select multiple>`
27. 标记语言: `import { marked } from 'marked';  <div>{@html marked(value)}</div>`
28. 生命周期: `onMount`, `beforeUpdate`准备数据, `afterUpdate`, `tick`批处理-等待 DOM 更新完再向下执行
29. 组件被销毁, 停止循环`return () => {};` [动画](https://learn.svelte.dev/tutorial/onmount)
30. [Eliza聊天机器人](https://learn.svelte.dev/tutorial/update)
31. `import { tick } from 'svelte';`, `await tick();`, 等待 DOM 更新完再向下执行
32. 可写存储: `Stores`订阅中心: `import { writable } from 'svelte/store';  export const count = writable(0);  count.update((n) => n + 1);`
33. [`Stores`订阅中心](https://learn.svelte.dev/tutorial/writable-stores)
34. 取消订阅: `import { onDestroy } from 'svelte';  onDestroy(unsubscribe);`
35. 自动订阅: 仅适用于在组件顶级范围声明（或导入）的存储变量, `<h1>The count is {$count}</h1>`
36. 可读存储: `readable(initValue, start, stop)`, 第一个订阅者时调用, 最后一个取消订阅时调用
37. 派生存储: `import { readable, derived } from 'svelte/store';`, 从多个输入存储中派生一个存储
38. 自定义存储: 减少定义别的组件
39. bind:存储