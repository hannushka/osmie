package missing_name.graph_connectivity;

import org.openstreetmap.atlas.geography.atlas.items.Edge;
import org.openstreetmap.atlas.tags.names.NameTag;

import java.util.Optional;

public class EdgeContainer  {
    public Edge edge;
    public long id;

    public EdgeContainer(Edge e) {
        edge = e;
    }

    public String getName() {
        Optional tmp = edge.getTag(NameTag.KEY);
        if (tmp.isPresent())
            return tmp.get().toString();
        return "";
    }

    @Override
    public boolean equals (Object other) {
        EdgeContainer e2 = (EdgeContainer) other;
        return Math.abs(edge.getIdentifier()) - Math.abs(e2.edge.getIdentifier()) == 0;
    }

    @Override
    public int hashCode() {
        Long tmp = (Long)Math.abs(edge.getIdentifier());
        return tmp.hashCode();
    }

}
