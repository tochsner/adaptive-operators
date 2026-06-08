package adapters;

import java.util.List;

public interface AdapterGenerator<T extends Adapter> {

    List<T> getAdapters();

}
