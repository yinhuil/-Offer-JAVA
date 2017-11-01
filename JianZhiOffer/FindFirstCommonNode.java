package JianZhiOffer;

public class FindFirstCommonNode {
	
//输入两个链表，找出它们的第一个公共结点。
	static class ListNode{
		int val;
		ListNode next;
		ListNode(int x){
			val=x;
		}
	}
	public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
		int lens1=findListLength(pHead1);
		int lens2=findListLength(pHead2);
		if(lens1>lens2){
			pHead1=walkStep(pHead1,lens1-lens2);
		}else{
			pHead2=walkStep(pHead2,lens2-lens1);
		}
		while(pHead1!=null){
			if(pHead1==pHead2) return pHead1;
			pHead1=pHead1.next;
			pHead2=pHead2.next;
		}
		return null;
	}
	private ListNode walkStep(ListNode pHead1, int i) {
		while(i>0){
			pHead1=pHead1.next;
			i--;
		}
		return pHead1;
	}
	private int findListLength(ListNode pHead1) {
		if(pHead1==null) return 0;
		int result=0;
		ListNode node=pHead1;
		while(node!=null){
			result++;
			node=node.next;
		}
		return result;
	}
}
