package JianZhiOffer;

public class match {
	
/*	题目描述
	请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'
	表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。
	在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"
	与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配*/
	
/*解题思路； 如果模式的后面一位不是*的情况，则直接是return matchCore(str, strIndex + 1, pattern, patternIndex + 1);
 * （相同就是上面的那个，不相同就是false），如果模式的后面以为是*的情况就会分为三种情况
 * 第一种： a 和a*，模式只取一个a，那就是 mathchcore(str,strIndex+1,pattern,patternIndex+2);
 * 第二种；a 和a*，模式一个a都不取 ，那就是matchcore(str, strIndex,pattern,patternIndex+2);
 * 第三种，a 和a*，模式会取多个a，那就是matchcore(str,strIndex+1,pattern,patternIndex)
 * 
 * 判定的条件就是，如果最后两个index都是相应的长度，则为true；
 * */
	public boolean match(char[] str, char[] pattern) {
		if (str == null || pattern == null) {
			return false;
		}
		int strIndex = 0;
		int patternIndex = 0;
		return matchCore(str, strIndex, pattern, patternIndex);
	}

	private boolean matchCore(char[] str, int strIndex, char[] pattern,
			int patternIndex) {
		if (strIndex == str.length && patternIndex == pattern.length) {
			return true;
		}
		if (strIndex != str.length && patternIndex == pattern.length) {
			return false;
		}
		if (patternIndex + 1 < pattern.length
				&& pattern[patternIndex + 1] == '*') {
			if ((strIndex != str.length && pattern[patternIndex] == str[strIndex])
					|| (pattern[patternIndex] == '.' && strIndex != str.length)) {

				return matchCore(str, strIndex, pattern, patternIndex + 2)
						|| matchCore(str, strIndex + 1, pattern,
								patternIndex + 2)
						|| matchCore(str, strIndex + 1, pattern, patternIndex);
			} else {
				return matchCore(str, strIndex, pattern, patternIndex + 2);
			}
		}

		if ((strIndex != str.length && pattern[patternIndex] == str[strIndex])
				|| (pattern[patternIndex] == '.' && strIndex != str.length)) {
			return matchCore(str, strIndex + 1, pattern, patternIndex + 1);
		}
		return false;
	}
}
