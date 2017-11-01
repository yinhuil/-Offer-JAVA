package leetcodeForJianzhi;

public class sortList {

/*Sort a linked list in O(n log n) time using constant space complexity*/
	
//	本题使用归并排序法
	static class ListNode{
		int val;
		ListNode next;
		ListNode(int x){
			val=x;
		}
	}
	 public ListNode sortList(ListNode head) {
	       if(head==null||head.next==null) return head;
			ListNode middle=getMiddle(head);
			ListNode right=sortList(middle.next);
			middle.next=null;
			ListNode left=sortList(head);
			return merge(left,right);
		}
		private ListNode merge(ListNode left, ListNode right) {
			ListNode start=new ListNode(0);
			ListNode last=start;
			if(left==null) return right;
			if(right==null) return left;
			while(left!=null && right!=null){
				if(left.val>right.val){
					last.next=right;
					right=right.next;
				}else{
					last.next=left;
					left=left.next;
				}
				last=last.next;
			}
			if(left==null){
				last.next=right;
			}
			if(right==null){
				last.next=left;
			}
			return start.next;
		}
		private ListNode getMiddle(ListNode head) {
			ListNode fast=head.next;
			ListNode slow=head;
			while(fast!=null && fast.next!=null){
				slow=slow.next;
				fast=fast.next.next;
			}
			return slow;
		}
}
