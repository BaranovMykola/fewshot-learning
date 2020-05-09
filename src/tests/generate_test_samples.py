def generate_test_samples(cats: int, samples: int):
    cats_id = [*range(cats)]

    def sample(n):
        return {
            'image': f'image_{n}',
            'mask': f'mask_{n}',
            'category_id': cats_id[n%cats],
            'id': n
        }

    def cat(n):
        return {
            'name': f'cat_{n}',
            'id': n
        }

    j = {
        'samples': [sample(x) for x in range(samples)],
        'categories': [cat(x) for x in cats_id]
    }

    return j
