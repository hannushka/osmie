package symspell;

import org.jetbrains.annotations.NotNull;

import java.util.Comparator;

public class SuggestItem implements Comparator<SuggestItem>, Comparable<SuggestItem>
{
    /// <summary>The suggested correctly spelled word.</summary>
    public String term;
    /// <summary>Edit distance between searched for word and suggestion.</summary>
    public int distance;
    /// <summary>Frequency of suggestion in the dictionary (a measure of how common the word is).</summary>
    public long count;

    /// <summary>Create a new instance of SuggestItem.</summary>
    /// <param name="term">The suggested word.</param>
    /// <param name="distance">Edit distance from search word.</param>
    /// <param name="count">Frequency of suggestion in dictionary.</param>
    public SuggestItem(String term, int distance, long count) {
        this.term = term;
        this.distance = distance;
        this.count = count;
    }

    @Override
    public int compare(SuggestItem suggestItem, SuggestItem t1) {
        return suggestItem.compareTo(t1);
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof SuggestItem && term.equals(((SuggestItem) obj).term);
    }

    @Override
    public int hashCode()
    {
        return term.hashCode();
    }

    @Override
    public String toString()
    {
        return "{" + term + ", " + distance + ", " + count + "}";
    }

    @Override
    public int compareTo(@NotNull SuggestItem other) {
        // order by distance ascending, then by frequency count descending
        if (this.distance == other.distance) return Long.compare(other.count, this.count);
        return Integer.compare(this.distance, other.distance);
    }

    public SuggestItem clone(){
        return new SuggestItem(this.term, this.distance, this.count);
    }
}
