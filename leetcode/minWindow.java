package leetcodeForJianzhi;
/*题目描述

Given a string S and a string T, find the minimum window in S which will 
contain all the characters in T in complexity O(n).
For example,
S ="ADOBECODEBANC"
T ="ABC"
Minimum window is"BANC".
Note: 
If there is no such window in S that covers all characters in T, 
return the emtpy string"".
If there are multiple such windows, you are guaranteed that there 
will always be only one unique minimum window in S.*/

//题解：这道题相当于滑窗口的方法。   首先将T里面的字符，存在map中，方便后边的使用，
//     令start=0，end=0，表示开始从第一个开始查找，如果发现s中的字符不在t中，则把map中的这个字符定义为-1，反之进行减一的操作，当count=0的时候则说明找到这样的字符串
//		，这个时候，就开始start++操作，表示开始向后移动，一直到第一次在T中的字符的后面，此时将count++，表示继续往前跑，直到end<S.length(),找到最小的字符串
public class minWindow {
	public static  String minWindow(String S, String T) {
		int[] map=new int[128];
		for(int i=0;i<T.length();i++){
			map[T.charAt(i)]++;
		}
		int start=0,end=0,count=T.length(),head=0,d=Integer.MAX_VALUE;
		while(end<S.length()){
			if(map[S.charAt(end++)]-->0){
				count--;
			}
//			要注意上面这行代码，表示如果此时字符出现在T中，则这个map值为0，并count++，如果不是，就为-1；
			while(count==0){
				if(end-start<d){
					d=end-(head=start);
					
				}
				if(map[S.charAt(start++)]++==0){
					count++;
				}
			}
		}
		return d==Integer.MAX_VALUE?"":S.substring(head, head+d);
	}
	public static void main(String[] args) {
		/*String s="ADOBECODEBANC";
		String t="ABC";
		System.out.println(minWindow(s, t));*/
		int temp=-1;
		if(temp++==0) System.out.println(1);
		else System.out.println(0);
	}
}
