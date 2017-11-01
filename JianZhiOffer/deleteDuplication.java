package JianZhiOffer;

public class deleteDuplication {
/*	在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，
	重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 
	处理后为 1->2->5*/
	
/*  题解：  （在链表的前面加一个值方法特别好）首先在链表的前面加一个-1的表头（这样的话我们就不需要找到第一个不是重复的数了）result，并result.next=pHead;
	然后将last=result
	组成一个新的链表，在接下的操作中，如果发现pHead.val==pHead.next.val就把pHead继续往下遍历,并将此时的pHead赋值给last.next
	如果发现不等于的话就把last专向下一个节点也就是last=pHead；
	*/
	
	static class ListNode {
		int val;
		ListNode next;

		ListNode(int x) {
			val = x;
		}
	}

	public static ListNode deleteDuplication(ListNode pHead) {
		ListNode result=new ListNode(0);
		ListNode pNode=pHead;
		result.next=pNode;
		ListNode last=result;
		while(pNode!=null && pNode.next!=null){
			if(pNode.val==pNode.next.val){
				int temp=pNode.val;
				while(pNode!=null && pNode.val==temp){
					pNode=pNode.next;
				}
				last.next=pNode;
			}else{
				last=pNode;
				pNode=pNode.next;
			}
		}
		return result.next;
	}
	public static void main(String[] args) {
		ListNode root1=new ListNode(1);
		ListNode root2=new ListNode(2);
		ListNode root3=new ListNode(3);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
		ListNode root4=new ListNode(3);
		ListNode root5=new ListNode(4);
		root1.next=root2;
		root2.next=root3;
		root3.next=root4;
		root4.next=root5;
		ListNode result=deleteDuplication(root1);
		while(result!=null){
			System.out.println(result.val);
			result=result.next;
		}
	}
}
