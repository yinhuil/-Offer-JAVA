package JianZhiOffer;

public class ReverseSentence42 {
	
	
/*	题目描述
	牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，
	写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，
	但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单
	词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序
	可不在行，你能帮助他么？*/
	public String ReverseSentence(String str) {
		if(str==null||str.trim().equals(""))return str;
		String result=reverse(str);
		String[] temp=result.split(" ");
		StringBuffer sb=new StringBuffer();
		for(int i=0;i<temp.length-1;i++){
			sb.append(reverse(temp[i])+" ");
		}
		sb.append(reverse(temp[temp.length-1]));
		return sb.toString().trim();
	}
	
	private static String reverse(String str) {
		String temp="";
		for(int i=0;i<str.length();i++){
			temp=str.charAt(i)+temp;
		}
		return temp;
	}
	
	public static void leftRotateString(String word,int n){
		String before=word.substring(0, n);
		String end=word.substring(n);
		before=reverse(before);
		end=reverse(end);
		
		before=before+end;
		before=reverse(before);
		System.out.println(before);
	}
	public static void main(String[] args) {
/*		String str="i am a student";
		reverSentence(str);*/
		String str="abcdefg";
		leftRotateString(str, 2);
		
		
	}
}
