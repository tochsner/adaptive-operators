package adaptiveoperators;

import org.junit.jupiter.api.Test;

import beast.base.inference.OperatorSchedule;

import java.lang.reflect.Field;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class GuidedMixedPreconditionedCrankNicolsonOperatorTest {

    @Test
    void deltaUsesLearnedCenterAndCovariance() throws ReflectiveOperationException {
        GuidedMixedPreconditionedCrankNicolsonOperator operator = newOperator(2);
        CenteredMultivariateNormalSampler sampler = sampler(operator);
        sampler.mean = new double[]{1.0, -1.0};

        sampler.record(new double[] {}, new double[]{3.0, -1.0});
        for (int i = 0; i < 63; i++) {
            sampler.record(new double[] {}, new double[]{1.0, 1.0});
        }

        double delta = operator.delta(new double[]{3.0, -1.0});

        assertThat(delta).isFinite();
        assertThat(delta).isGreaterThan(0.0);
    }

    @Test
    void logReferenceDensityMatchesHaarMixtureReference() throws ReflectiveOperationException {
        GuidedMixedPreconditionedCrankNicolsonOperator operator = newOperator(4);

        assertThat(operator.logReferenceDensity(9.0))
                .isCloseTo(-2.0 * Math.log(9.0), within(1.0E-12));
    }

    @Test
    void rejectFlipsSelectedDirection() throws ReflectiveOperationException {
        GuidedMixedPreconditionedCrankNicolsonOperator operator = newOperator(1);
        Field selectedDirectionIdx = GuidedMixedPreconditionedCrankNicolsonOperator.class
                .getDeclaredField("selectedDirectionIdx");
        selectedDirectionIdx.setAccessible(true);
        selectedDirectionIdx.setInt(operator, 0);

        operator.reject(0);

        int[] directions = directions(operator);
        assertThat(directions[0]).isEqualTo(-1);
    }

    private static GuidedMixedPreconditionedCrankNicolsonOperator newOperator(int dimension)
            throws ReflectiveOperationException {
        GuidedMixedPreconditionedCrankNicolsonOperator operator = new GuidedMixedPreconditionedCrankNicolsonOperator();
        operator.setOperatorSchedule(new OperatorSchedule());
        setField(operator, "sampler", new CenteredMultivariateNormalSampler(dimension));
        setField(operator, "directions", new int[]{1});
        return operator;
    }

    private static CenteredMultivariateNormalSampler sampler(GuidedMixedPreconditionedCrankNicolsonOperator operator)
            throws ReflectiveOperationException {
        Field field = GuidedMixedPreconditionedCrankNicolsonOperator.class.getDeclaredField("sampler");
        field.setAccessible(true);
        return (CenteredMultivariateNormalSampler) field.get(operator);
    }

    private static int[] directions(GuidedMixedPreconditionedCrankNicolsonOperator operator)
            throws ReflectiveOperationException {
        Field field = GuidedMixedPreconditionedCrankNicolsonOperator.class.getDeclaredField("directions");
        field.setAccessible(true);
        return (int[]) field.get(operator);
    }

    private static void setField(Object target, String fieldName, Object value) throws ReflectiveOperationException {
        Field field = GuidedMixedPreconditionedCrankNicolsonOperator.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

}
