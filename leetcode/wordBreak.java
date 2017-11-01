package leetcodeForJianzhi;

import java.util.Set;

public class wordBreak {
/*	题目描述

	Given a string s and a dictionary of words dict, determine if s can be
	segmented into a space-separated sequence of one or more dictionary words.
	For example, given
	s ="leetcode",
	dict =["leet", "code"].
	Return true because"leetcode"can be segmented as"leet code".*/	
	
////解析：用一个s.length()长度的boolean类型数组，   先定一个位标i，  然后来定一个j从0一直到i，如果发现s.substring(j, i + 1)包含在dict里面
//	就可以令其为true，这里面一定要注意d[i] = d[i] || (d[j - 1] && subFlag);中的d[i] 项，因为我们后面继续的时候需要把以前的
//	数据保存以来，有可能前面有出现true的情况。
	public boolean wordBreak(String s, Set<String> dict) {
		int len = s.length();
		boolean[] d = new boolean[len];
		for (int k = 0; k < len; k++)
			d[k] = false;
		String sub = null;
		boolean subFlag = false;

		for (int i = 0; i < len; i++) {
			for (int j = 0; j <= i; j++) {
				sub = s.substring(j, i + 1);
				subFlag = dict.contains(sub);
				if (j == 0)
					d[i] = subFlag;
				else
					d[i] = d[i] || (d[j - 1] && subFlag);
			}
		}

		return d[len - 1];
	}

}
