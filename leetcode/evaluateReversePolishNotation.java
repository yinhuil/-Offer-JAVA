package leetcodeForJianzhi;

import java.util.Stack;

/*题目描述

Evaluate the value of an arithmetic expression in Reverse Polish Notation.
Valid operators are+,-,*,/. Each operand may be an integer or another expression.
Some examples:
  ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9
  ["4", "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6*/
public class evaluateReversePolishNotation {
	public static int evalRPN(String[] tokens) {
		if(tokens.length==0) return 0;
		Stack<Integer> sta=new Stack<Integer>();
		for(int i=0;i<tokens.length;i++){
			if(tokens[i]!="+" &&tokens[i]!="-"&&tokens[i]!="*"&&tokens[i]!="/"){
				sta.push(Integer.parseInt(tokens[i]));
			}else{
				int temp1=sta.pop();
				int temp2=sta.pop();
				if(tokens[i]=="+"){
					sta.push(temp1+temp2);
				}else if(tokens[i]=="-"){
					sta.push(temp2-temp1);
				}else if (tokens[i]=="*"){
					sta.push(temp2*temp1);
				}else{
					sta.push(temp2/temp1);
				}
			}
		}
		return sta.pop();
	}
	public static void main(String[] args) {
		String[] str={"0", "1", "/"};
		System.out.println(evalRPN(str));
	}
}
