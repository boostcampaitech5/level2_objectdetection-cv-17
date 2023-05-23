class GaussianNoise:
    def __init__(self, mean=0, std=1, prob=0.5):
        self.mean = mean
        self.std = std
        self.prob = prob

    def _apply_gaussian_noise(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            noise = np.random.normal(self.mean, self.std, img.shape).astype(np.uint8)
            noisy_img = np.clip(img + noise, 0, 255)
            results[key] = noisy_img

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        self._apply_gaussian_noise(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, prob={self.prob})'
        return repr_str
    
    ## augmentation 정의 후 
    ## mmdetection/mmdet/datasets/pipelines/__init__.py 에 추가