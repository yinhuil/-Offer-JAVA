package JianZhiOffer;


public class ReverseList16 {
//	输入一个链表，反转链表后，输出链表的所有元素。
	
/*	解析；先定下一个节点pNext，再把当前节点与下一个节点断开 pHead.next=pre，  在把 pHead赋值给pre，最后吧pNext赋值给pHead
 * 方便下一次的递归
 * 
 * */
	static class ListNode{
		int val;
		ListNode next;
		ListNode(int x){
			val=x;
		}
	}
	public ListNode ReverseList(ListNode head) {
		ListNode pReverse=null;
		ListNode pHead=head;
		ListNode pre=null;
		while(pHead!=null){
			ListNode pNext=pHead.next;
			if(pNext==null){
				pReverse=pHead;
			}
			pHead.next=pre;
			pre=pHead;
			pHead=pNext;
		}
		return pReverse;
	}
	
}
