package JianZhiOffer;

import java.util.LinkedHashMap;
//有hashmap来存储字符
public class FirstNotRepeatChar35 {
	public static Character firstNotRepeating(String str) {
		if (str == null)
			return null;
		char[] strChar = str.toCharArray();
		LinkedHashMap<Character, Integer> hash = new LinkedHashMap<Character, Integer>();
		for (char item : strChar) {
			if (hash.containsKey(item))
				hash.put(item, hash.get(item) + 1);
			else
				hash.put(item, 1);
		}
		for (char key : hash.keySet()) {
			if (hash.get(key) == 1)
				return key;
		}
		return null;
	} 

	public int FirstNotRepeatingChar(String str) {
		char[] chars = str.toCharArray();
		int[] map = new int[256];
		for (int i = 0; i < chars.length; i++) {
			map[chars[i]]++;
		}
		for (int i = 0; i < chars.length; i++) {
			if (map[chars[i]] == 1)
				return i;
		}
		return -1;
	}

	public static void main(String[] args) {
		String str = "abaccdebff";   
        System.out.println(firstNotRepeating(str));  
	}

}
