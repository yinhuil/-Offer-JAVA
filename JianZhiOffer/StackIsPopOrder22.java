package JianZhiOffer;

import java.util.Stack;

public class StackIsPopOrder22 {

/*	题目描述
	输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。
	假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4，5,3,2,1
	是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。
	（注意：这两个序列的长度是相等的）	*/
	
	
	
//	第一种方法更好理解，  用一个堆栈来模拟环境，判断最后堆栈是否为空，如果为空的话，就是按照popA的方式全部吐出来了。
	public static boolean IsPopOrder(int[] pushA, int[] popA) {
		if(pushA==null || popA==null) return false;
		Stack<Integer> sta=new Stack<Integer>();
		int indepop=0;
		for(int i=0;i<pushA.length;i++){
			sta.push(pushA[i]);
			while(!sta.isEmpty() && sta.peek()==popA[indepop]){
				sta.pop();
				indepop++;
			}
		}
		return sta.isEmpty();
	}
	
	
	
	
	
	
//	第二种方法   其实是一致的
	public static boolean isPopOrder(int[] pushOrder,int[] popOrder,int length) {
		boolean possible=false;
		if(pushOrder!=null && popOrder!=null&& length>=0){
			Stack<Integer> sta=new Stack<Integer>();
			int pushId=0;
			int popId=0;
			while(popId<length){
				while(sta.empty() || sta.peek()!=popOrder[popId]){
					if(pushId==length){
						break;
					}
					sta.push(pushOrder[pushId++]);
				}
				if(sta.peek()!=popOrder[popId]){
					break;
				}
				sta.pop();
				popId++;
			}
			if(sta.empty() && popId==length){
				possible=true;
			}
		}
		return possible;
	}
	public static void main(String[] args) {
		int[] push={1,2,3,4,5};
		int[] pop={4,5,3,2,1};
		System.out.println(isPopOrder(push, pop, 5));
	}
}
