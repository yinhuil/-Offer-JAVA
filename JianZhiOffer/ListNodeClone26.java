package JianZhiOffer;

public class ListNodeClone26 {
	static class ListNode {
		int value;
		ListNode next;
		ListNode sibling;
//		ListNode(int x) {
//			value = x;
//		}
	}

	public static void cloneNode(ListNode pHead){
		ListNode pNode=pHead;
		while(pNode!=null){
			ListNode pClone=new ListNode();
			pClone.value=pNode.value;
			pClone.next=pNode.next;
			pClone.sibling=null;
			pNode.next=pClone;
			pNode=pClone.next;
		}
	}
	public static void connectSibling(ListNode pHead){
		ListNode pNode=pHead;
		while(pNode!=null){
			ListNode pclone=pNode.next;
			if(pNode.sibling!=null){
				pclone.sibling=pNode.sibling.next;
			}
			pNode=pclone.next;
		}
	}
	public static ListNode reconnectNodes(ListNode pHead){
		ListNode pNode=pHead;
		ListNode cloneHead=null;
		ListNode cloneNode=null;
		if(pNode!=null){
			cloneHead=cloneNode=pNode.next;
			pNode.next=cloneHead.next;
			pNode=pNode.next;
		}
		while(pNode!=null){
			cloneNode.next=pNode.next;
			cloneNode=cloneNode.next;
			pNode.next=cloneNode.next;
			pNode=pNode.next;
			
		}
		return cloneHead;	
	}
	public static ListNode clone(ListNode head){
		if(head==null){
			return null;
		}
		cloneNode(head);
		connectSibling(head);
		return reconnectNodes(head);
	}
	public static void printList(ListNode head){
		while(head!=null){
			System.out.println(head.value+"->");
			head=head.next;
		}
		System.out.println("null");
	}
	public static boolean isSame(ListNode head1,ListNode head2){
		while(head1!=null && head2!=null){
			if(head1==head2){
				head1=head1.next;
				head2=head2.next;
			}else{
				return false;
			}
		}
		return head1 == null && head2 == null;
	}
	public static void main(String[] args) {
		ListNode  head = new ListNode();
        head.value = 1;
        head.next = new ListNode();
        head.next.value = 2;
        head.next.next = new ListNode();
        head.next.next.value = 3;
        head.next.next.next = new ListNode();
        head.next.next.next.value = 4;
        head.next.next.next.next = new ListNode();
        head.next.next.next.next.value = 5;

        head.sibling = head.next.next;
        head.next.sibling = head.next.next.next.next.next;
        head.next.next.next.sibling = head.next;

        ListNode tmp = head;
        printList(head);
        ListNode newHead = clone(head);
        printList(head);
        System.out.println(isSame(head, tmp));
        printList(newHead);
        System.out.println(isSame(head, newHead));
	}
}
