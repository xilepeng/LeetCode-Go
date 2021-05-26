




### LRUCache

```go
type LRUCache struct {
	cache          map[int]*DLinkedNode
	head, tail     *DLinkedNode
	size, capacity int
}

type DLinkedNode struct {
	key, value int
	prev, next *DLinkedNode
}

func initDLinkedNode(key, value int) *DLinkedNode {
	return &DLinkedNode{
		key:   key,
		value: value,
	}
}

func Constructor(capacity int) LRUCache {
	l := LRUCache{
		cache:    map[int]*DLinkedNode{},
		head:     initDLinkedNode(0, 0),
		tail:     initDLinkedNode(0, 0),
		capacity: capacity,
	}
	l.head.next = l.tail
	l.tail.prev = l.head
	return l
}

func (this *LRUCache) Get(key int) int {
	if _, ok := this.cache[key]; !ok {
		return -1
	}
	node := this.cache[key] // 如果 key 存在，先通过哈希表定位，再移到头部
	this.moveToHead(node)
	return node.value
}

func (this *LRUCache) Put(key int, value int) {
	if _, ok := this.cache[key]; !ok { // 如果 key 不存在，创建一个新的节点
		node := initDLinkedNode(key, value)
		this.cache[key] = node // 添加进哈希表
		this.addToHead(node)   // 添加至双向链表的头部
		this.size++
		if this.size > this.capacity {
			removed := this.removeTail()    // 如果超出容量，删除双向链表的尾部节点
			delete(this.cache, removed.key) // 删除哈希表中对应的项
			this.size--
		}
	} else { // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
		node := this.cache[key]
		node.value = value
		this.moveToHead(node)
	}
}

func (this *LRUCache) addToHead(node *DLinkedNode) {
	node.prev = this.head
	node.next = this.head.next
	this.head.next.prev = node
	this.head.next = node
}

func (this *LRUCache) removeNode(node *DLinkedNode) {
	node.prev.next = node.next
	node.next.prev = node.prev
}

func (this *LRUCache) moveToHead(node *DLinkedNode) {
	this.removeNode(node)
	this.addToHead(node)
}

func (this *LRUCache) removeTail() *DLinkedNode {
	node := this.tail.prev
	this.removeNode(node)
	return node
}
```

## Sorting

### 1. [Quick Sort](https://www.hackerearth.com/practice/algorithms/sorting/quick-sort/tutorial/)


快速排序基于分而治之的方法，随机选择枢轴元素划分数组，左边小于枢轴、右边大于枢轴，递归处理左右两边

```go
func quick_sort(A []int, start, end int) {
	if start < end {
		piv_pos := random_partition(A, start, end)
		quick_sort(A, start, piv_pos-1)
		quick_sort(A, piv_pos+1, end)
	}
}
func partition(A []int, start, end int) int {
	piv, i := A[start], start+1//第一个元素作为枢轴
	for j := start + 1; j <= end; j++ {
		if A[j] < piv {//小于枢轴的放一边、大于枢轴的放另一边
			A[i], A[j] = A[j], A[i]
			i++
		}
	}
	A[start], A[i-1] = A[i-1], A[start] //放置枢轴到正确的位置
	return i - 1 						//返回枢轴的位置
}
func random_partition(A []int, start, end int) int {
	rand.Seed(time.Now().Unix())
	random := start + rand.Int()%(end-start+1)
	A[start], A[random] = A[random], A[start]
	return partition(A, start, end)
}
```




### 2. [Heap Sort](https://www.hackerearth.com/practice/algorithms/sorting/heap-sort/tutorial/)

在大根堆中、最大元素总在根上，堆排序使用堆的这个属性进行排序

```go
func heap_sort(A []int) {
	heap_size := len(A)
	build_maxheap(A, heap_size)
	for i := heap_size - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]      //交换堆顶与堆底元素，最大值放置在数组末尾
		heap_size--                  //剩余待排序元素整理成堆
		max_heapify(A, 0, heap_size) //堆顶 root 向下调整
	}
}
func build_maxheap(A []int, heap_size int) {
	for i := heap_size >> 1; i >= 0; i-- { // heap_size / 2 后面都是叶子节点，不需要向下调整
		max_heapify(A, i, heap_size)
	}
}
func max_heapify(A []int, i, heap_size int) {
	lson, rson, largest := i<<1+1, i<<1+2, i
	for lson < heap_size && A[largest] < A[lson] { //左儿子存在并大于根
		largest = lson
	}
	for rson < heap_size && A[largest] < A[rson] { //右儿子存在并大于根
		largest = rson
	}
	if i != largest { //找到左右儿子的最大值
		A[i], A[largest] = A[largest], A[i] //堆顶调整为最大值
		max_heapify(A, largest, heap_size)  //递归调整子树
	}
}
```



### 3. [Merge Sort](https://www.hackerearth.com/practice/algorithms/sorting/merge-sort/tutorial/)

归并排序是一种分而治之的算法，其思想是将一个列表分解为几个子列表，直到每个子列表由一个元素组成，然后将这些子列表合并为排序后的列表。

```go
func merge_sort(A []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1 //分2部分定义当前数组
		merge_sort(A, start, mid)     //排序数组的第1部分
		merge_sort(A, mid+1, end)     //排序数组的第2部分
		merge(A, start, mid, end)     //通过比较2个部分的元素来合并2个部分
	}
}
func merge(A []int, start, mid, end int) {
	Arr := make([]int, end-start+1)
	p, q, k := start, mid+1, 0
	for i := start; i <= end; i++ {
		if p > mid { //检查第一部分是否到达末尾
			Arr[k] = A[q]
			q++
		} else if q > end { //检查第二部分是否到达末尾
			Arr[k] = A[p]
			p++
		} else if A[p] <= A[q] { //检查哪一部分有更小的元素
			Arr[k] = A[p]
			p++
		} else {
			Arr[k] = A[q]
			q++
		}
		k++
	}
	// copy(A[start:end+1], Arr)
	for p := 0; p < k; p++ {
		A[start] = Arr[p]
		start++
	}
}
```




### 4. [Insertion Sort](https://www.hackerearth.com/practice/algorithms/sorting/insertion-sort/tutorial/#c252800)

插入排序基于这样的想法：每次迭代都会消耗输入元素中的一个元素，以找到其正确位置，即该元素在排序数组中的位置。

通过在每次迭代时增加排序后的数组来迭代输入元素。它将当前元素与已排序数组中的最大值进行比较。如果当前元素更大，则它将元素留在其位置，然后移至下一个元素，否则它将在已排序数组中找到其正确位置，并将其移至该位置。这是通过将已排序数组中所有大于当前元素的元素移动到前面的一个位置来完成的

```go
func insertion_sort(A []int, n int) {
	for i := 0; i < n; i++ {
		temp, j := A[i], i
		for j > 0 && temp < A[j-1] { //当前元素小于左边元素
			A[j] = A[j-1] //向前移动左边元素->
			j--
		}
		A[j] = temp //移动当前元素到正确的位置
	}
}
```





### 5. [Bubble Sort](https://www.hackerearth.com/practice/algorithms/sorting/bubble-sort/tutorial/)


反复比较成对的相邻元素，交换它们的位置如果他们在无序区。（最大元素冒泡到最后）

```go
func bubble_sort(A []int, n int) {
	for k := 0; k < n-1; k++ {  // (n-k-1) 是忽略比较的元素，这些元素已比较完成在简单的迭代中
		for i := 0; i < n-k-1; i++ {
			if A[i] > A[i+1] {
				A[i], A[i+1] = A[i+1], A[i] //交换
			}
		}
	}
}
```


### 6. [Selection Sort](https://www.hackerearth.com/practice/algorithms/sorting/selection-sort/tutorial/)

在未排序的数组中找到最小或最大元素，然后将其放在已排序的数组中的正确位置。

```go
func selection_sort(A []int, n int) {
	for i := 0; i < n-1; i++ {		 //在每次迭代中将数组的有效大小减少1
		min := i                     //假设第一个元素是未排序数组的最小值
		for j := i + 1; j < n; j++ { //给出未排序数组的有效大小
			if A[j] < A[min] { //找到最小的元素
				min = j
			}
		}
		A[i], A[min] = A[min], A[i] //将最小元素放在适当的位置
	}
}
```


欢迎大家加入  go早起刷题打卡群 


![群二维码地址](http://ww1.sinaimg.cn/large/007daNw2ly1gqvm542i7oj30fo0luwhf.jpg)

![过期点我](http://ww1.sinaimg.cn/large/007daNw2ly1gqvm5w0rjvj30fo0lu41c.jpg)
