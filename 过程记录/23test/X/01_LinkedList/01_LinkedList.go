package _1_LinkedList

type ListNode struct {
	value interface{}
	next  *ListNode
}

type LinkedList struct {
	head   *ListNode
	length uint
}

func NewListNode(v interface{}) *ListNode {
	return &ListNode{v, nil}
}

func (this *ListNode) GetNext() *ListNode {
	return this.next
}

func (this *ListNode) GetValue() interface{} {
	return this.value
}

func NewLinkedList() *LinkedList {
	return &LinkedList{NewListNode(0), 0}
}

// InsertAfter 在某个节点后面插入节点
func (this *LinkedList) InsertAfter(p *ListNode, v interface{}) bool {
	if nil == p {
		return false
	}
	newNode := NewListNode(v)
	oldNext := p.next
	p.next = newNode
	newNode.next = oldNext
	this.length++
	return true
}

// InsertBefore 在某个节点前面插入节点
func (this *LinkedList) InsertBefore(p *ListNode, v interface{}) bool {
	if nil == p || p == this.head {
		return false
	}
	prev := this.head
	curr := this.head.next
	for curr != nil {
		if curr == p {
			break
		}
		prev = curr
		curr = curr.next
	}
	if nil == curr {
		return false
	}
	newNode := NewListNode(v)
	prev.next = newNode
	newNode.next = curr
	this.length++
	return true
}

// Reverse 反转链表
func Reverse(head *ListNode) *ListNode {
	var prev *ListNode
	curr := head
	for curr != nil {
		succ := curr.next
		curr.next = prev
		prev = curr
		curr = succ
	}
	return prev
}
